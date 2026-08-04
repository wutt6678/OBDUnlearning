from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    logging as hf_logging,
)


hf_logging.set_verbosity_error()

def tied_weights_mapping(value):
    if value is None:
        return {}
    if hasattr(value, "keys"):
        return value
    if isinstance(value, (list, tuple, set)):
        return {item: item for item in value}
    return {}


def normalize_tied_weights_keys(model) -> None:
    value = getattr(model, "all_tied_weights_keys", None)
    if value is not None and not hasattr(value, "keys"):
        try:
            model.all_tied_weights_keys = tied_weights_mapping(value)
        except Exception:
            setattr(model.__class__, "all_tied_weights_keys", tied_weights_mapping(value))


def install_transformers_compat_shims() -> None:
    """Patch small API gaps needed by some remote model files."""
    import transformers.utils as transformers_utils
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(transformers_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            pass

        transformers_utils.LossKwargs = LossKwargs

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        def get_all_tied_weights_keys(self):
            value = self.__dict__.get("_obd_all_tied_weights_keys", None)
            if value is None:
                value = getattr(self, "_tied_weights_keys", None)
            return tied_weights_mapping(value)

        def set_all_tied_weights_keys(self, value):
            self.__dict__["_obd_all_tied_weights_keys"] = value

        PreTrainedModel.all_tied_weights_keys = property(
            get_all_tied_weights_keys,
            set_all_tied_weights_keys,
        )

    try:
        import transformers.integrations.accelerate as accelerate_integration
    except Exception:
        return
    if not getattr(accelerate_integration.compute_module_sizes, "_obd_tied_patch", False):
        original_compute_module_sizes = accelerate_integration.compute_module_sizes

        def compute_module_sizes_with_tied_key_mapping(model, *args, **kwargs):
            normalize_tied_weights_keys(model)
            return original_compute_module_sizes(model, *args, **kwargs)

        compute_module_sizes_with_tied_key_mapping._obd_tied_patch = True
        accelerate_integration.compute_module_sizes = compute_module_sizes_with_tied_key_mapping

    init_device_map = getattr(accelerate_integration, "_init_infer_auto_device_map", None)
    if init_device_map is not None and not getattr(init_device_map, "_obd_tied_patch", False):
        original_init_device_map = init_device_map

        def init_infer_auto_device_map_with_tied_key_mapping(model, *args, **kwargs):
            normalize_tied_weights_keys(model)
            return original_init_device_map(model, *args, **kwargs)

        init_infer_auto_device_map_with_tied_key_mapping._obd_tied_patch = True
        accelerate_integration._init_infer_auto_device_map = init_infer_auto_device_map_with_tied_key_mapping


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def load_model_and_tokenizer(cfg: dict):
    install_transformers_compat_shims()
    model_name = cfg["name_or_path"]
    tokenizer_name = cfg.get("tokenizer_name_or_path", model_name)
    trust_remote_code = cfg.get("trust_remote_code", True)
    adapter_path = Path(model_name)
    is_adapter = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()
    if is_adapter and "tokenizer_name_or_path" not in cfg:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if cfg.get("device", "auto") == "auto" else None
    base_model_name = model_name
    if is_adapter:
        base_model_name = PeftConfig.from_pretrained(model_name).base_model_name_or_path

    model_config = AutoConfig.from_pretrained(
        base_model_name,
        trust_remote_code=trust_remote_code,
    )
    model_trust_remote_code = trust_remote_code
    # ChatGLM's custom modeling code requires config.max_length at __init__.
    _patched_max_length = False
    if not hasattr(model_config, "max_length"):
        seq_length = getattr(model_config, "seq_length", None)
        if seq_length is not None:
            model_config.max_length = seq_length
            _patched_max_length = True
            print(f"patched config.max_length={seq_length} from seq_length")
    if not hasattr(model_config, "use_cache"):
        model_config.use_cache = False
        print("patched config.use_cache=False")
    if not hasattr(model_config, "num_hidden_layers"):
        num_layers = getattr(model_config, "num_layers", None)
        if num_layers is not None:
            model_config.num_hidden_layers = num_layers
            print(f"patched config.num_hidden_layers={num_layers} from num_layers")

    if model_config.model_type == "phi3" and not cfg.get("force_remote_code", False):
        # Phi-4-mini remote code is currently incompatible with this Transformers
        # version's tied-weight API. The native Phi3 implementation is available
        # and avoids that remote-code mismatch.
        model_trust_remote_code = False
        print("trust_remote_code=true ignored for phi3 model loading; set force_remote_code=true to use remote model code")

    if model_config.model_type == "mistral3":
        model_loader = AutoModelForImageTextToText
    elif model_config.model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_6_moe"}:
        requested_multimodal = cfg.get("use_multimodal_loader", False)
        force_multimodal = cfg.get("force_multimodal_loader", False)
        if requested_multimodal and not force_multimodal:
            print("use_multimodal_loader=true ignored for Qwen text fine-tuning; set force_multimodal_loader=true for vision-language runs")
        model_loader = AutoModelForMultimodalLM if requested_multimodal and force_multimodal else AutoModelForCausalLM
    else:
        model_loader = AutoModelForCausalLM
    model_kwargs = {
        "config": model_config,
        "dtype": resolve_dtype(cfg.get("dtype", "auto")),
        "device_map": device_map,
        "trust_remote_code": model_trust_remote_code,
    }
    if cfg.get("max_memory") is not None:
        model_kwargs["max_memory"] = cfg["max_memory"]
    print(f"model loader: {model_loader.__name__} for model_type={model_config.model_type}")
    model = model_loader.from_pretrained(
        base_model_name,
        **model_kwargs,
    )
    print(f"model class: {model.__class__.__name__}")
    # Remove patched max_length so newer transformers doesn't reject it during generation.
    if _patched_max_length and hasattr(model.config, "max_length"):
        del model.config.max_length
    # Newer transformers removed _extract_past_from_model_output which some custom
    # model code (e.g. ChatGLM) still calls in _update_model_kwargs_for_generation.
    if not hasattr(model, "_extract_past_from_model_output"):
        def _extract_past_from_model_output(self, outputs, standardize_cache_format=True):
            return getattr(outputs, "past_key_values", None)
        import types
        model._extract_past_from_model_output = types.MethodType(_extract_past_from_model_output, model)
    if hasattr(model, "hf_device_map"):
        print(f"hf_device_map: {model.hf_device_map}")
    else:
        print(f"model device: {next(model.parameters()).device}")
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        model.config.text_config.pad_token_id = tokenizer.pad_token_id
        if getattr(model.config.text_config, "eos_token_id", None) is None:
            model.config.text_config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if getattr(model.generation_config, "eos_token_id", None) is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id

    if is_adapter:
        model = PeftModel.from_pretrained(model, model_name, is_trainable=cfg.get("use_lora", True))
        model.print_trainable_parameters()
    elif cfg.get("use_lora", True):
        lora_cfg = cfg.get("lora", {})
        targets = lora_cfg.get("target_modules", "auto")
        if targets == "auto":
            targets = infer_lora_targets(model)
        print(f"LoRA target modules: {targets}")
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=targets,
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def has_multimodal_config(model_config) -> bool:
    multimodal_attrs = (
        "vision_config",
        "visual_config",
        "image_token_id",
        "video_token_id",
        "mm_projector_type",
    )
    return any(getattr(model_config, attr, None) is not None for attr in multimodal_attrs)


def infer_lora_targets(model) -> list[str]:
    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "fc1",
        "fc2",
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    }
    excluded_parts = {
        "vision_tower",
        "vision_model",
        "visual",
        "multi_modal_projector",
        "mm_projector",
        "projector",
    }
    targets = []
    for module_name, module in model.named_modules():
        if type(module) is not torch.nn.Linear:
            continue
        parts = module_name.split(".")
        if any(part in excluded_parts for part in parts):
            continue
        if parts[-1] in candidates:
            targets.append(module_name)
        elif len(parts) >= 2 and parts[-2] in candidates and parts[-1] == "linear":
            targets.append(module_name)
    if not targets:
        raise ValueError("Could not infer supported LoRA target modules for this model.")
    return targets


def model_device(model) -> torch.device:
    return next(model.parameters()).device
