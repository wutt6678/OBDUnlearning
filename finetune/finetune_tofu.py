from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import CPUOffload, FullStateDictConfig, MixedPrecision, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import set_seed

from src.config import load_config
from src.data import TofuExample, load_config_examples
from src.models import load_model_and_tokenizer, model_device


PLAIN_PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
PLAIN_MCQ_PROMPT_TEMPLATE = "Question: {question}\nAnswer:"


def distributed_requested(train_cfg: dict) -> bool:
    return train_cfg.get("distributed") == "fsdp" or int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(train_cfg: dict) -> tuple[bool, int, int, int, torch.device]:
    is_distributed = distributed_requested(train_cfg)
    if not is_distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device

    if not torch.cuda.is_available():
        raise RuntimeError("FSDP fine-tuning requires CUDA GPUs.")
    if not dist.is_initialized():
        dist.init_process_group(backend=train_cfg.get("dist_backend", "nccl"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return True, rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rank0_print(is_main: bool, *args, **kwargs) -> None:
    if is_main:
        print(*args, **kwargs)


def fsdp_param_dtype(train_cfg: dict):
    dtype_name = train_cfg.get("fsdp_mixed_precision", "bfloat16")
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "none": None,
    }.get(dtype_name)


def cast_floating_params(model, target_dtype, is_main: bool) -> None:
    if target_dtype is None:
        return
    converted = 0
    total = 0
    with torch.no_grad():
        for param in model.parameters():
            if not param.is_floating_point():
                continue
            total += 1
            if param.dtype != target_dtype:
                param.data = param.data.to(dtype=target_dtype)
                converted += 1
    rank0_print(
        is_main,
        f"FSDP param dtype alignment: converted {converted}/{total} floating params to {target_dtype}",
    )


def wrap_fsdp(model, train_cfg: dict, device: torch.device, is_main: bool = True):
    min_params = int(train_cfg.get("fsdp_min_num_params", 1_000_000))
    dtype = fsdp_param_dtype(train_cfg)
    mixed_precision = None
    if dtype is not None:
        mixed_precision = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )
    cpu_offload = None
    if train_cfg.get("fsdp_cpu_offload", False):
        cpu_offload = CPUOffload(offload_params=True)

    ignored_modules = []
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    for module in (input_embeddings, output_embeddings):
        if module is not None and module not in ignored_modules:
            ignored_modules.append(module)
    rank0_print(
        is_main,
        f"FSDP ignored modules: {[type(module).__name__ for module in ignored_modules]}",
    )
    for module in ignored_modules:
        module.to(device)

    wrap_policy = train_cfg.get("fsdp_wrap_policy", "transformer_block")
    if wrap_policy == "size_based":
        auto_wrap_policy = lambda module, recurse, nonwrapped_numel: size_based_auto_wrap_policy(
            module,
            recurse,
            nonwrapped_numel,
            min_num_params=min_params,
        )
    elif wrap_policy == "transformer_block":
        auto_wrap_policy = lambda module, recurse, nonwrapped_numel: True if recurse else is_transformer_block(module)
    else:
        raise ValueError(f"Unsupported fsdp_wrap_policy: {wrap_policy}")
    rank0_print(is_main, f"FSDP wrap policy: {wrap_policy}")
    rank0_print(is_main, "FSDP use_orig_params: True")

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        ignored_modules=ignored_modules,
        device_id=device,
        backward_prefetch=None,
        use_orig_params=True,
    )


def is_transformer_block(module) -> bool:
    name = type(module).__name__.lower()
    has_attention = (
        hasattr(module, "self_attn")
        or hasattr(module, "self_attention")
        or hasattr(module, "attention")
    )
    has_mlp = hasattr(module, "mlp") or hasattr(module, "feed_forward")
    return has_attention and has_mlp and ("layer" in name or "block" in name)


def enable_memory_saving(model, train_cfg: dict, is_main: bool) -> None:
    if train_cfg.get("gradient_checkpointing", True):
        # Use FSDP-safe non-reentrant activation checkpointing instead of
        # HuggingFace model.gradient_checkpointing_enable(), which can trigger
        # inconsistent FSDP all-gather order for some decoder-only models.
        wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=wrapper,
            check_fn=is_transformer_block,
        )
        rank0_print(is_main, "activation checkpointing: enabled with FSDP NO_REENTRANT wrapper")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
        rank0_print(is_main, "model.config.use_cache: false")


class TofuSFTDataset(Dataset):
    def __init__(self, examples: list[TofuExample], tokenizer, max_length: int, prompt_format: str):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_format = prompt_format

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt, full_text = format_example(
            tokenizer=self.tokenizer,
            question=ex.question,
            answer=ex.answer,
            prompt_format=self.prompt_format,
        )
        enc, labels = encode_supervised_text(
            tokenizer=self.tokenizer,
            prompt=prompt,
            full_text=full_text,
            max_length=self.max_length,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "question": ex.question,
            "answer": ex.answer,
            "profile_id": ex.profile_id,
        }


def encode_supervised_text(tokenizer, prompt: str, full_text: str, max_length: int) -> tuple[dict, list[int]]:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_tensors=None,
    )["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_tensors=None,
    )
    input_ids = list(full["input_ids"])
    prompt_len = min(len(prompt_ids), len(input_ids))

    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        prompt_len = max(prompt_len - overflow, 0)

    labels = list(input_ids)
    labels[:prompt_len] = [-100] * prompt_len
    if all(label == -100 for label in labels):
        # Left-truncation removed all supervised tokens (prompt >= max_length).
        # Fall back to right-truncation to preserve at least some answer tokens.
        input_ids = list(full["input_ids"])[:max_length]
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = list(input_ids)
        labels[:prompt_len] = [-100] * prompt_len
        if all(label == -100 for label in labels):
            # Prompt alone exceeds max_length; force-supervise the last token
            # so the sample is valid but contributes minimal loss.
            labels[-1] = input_ids[-1]

    enc = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
    }
    return enc, labels


def format_example(tokenizer, question: str, answer: str, prompt_format: str) -> tuple[str, str]:
    if prompt_format == "plain_qa":
        prompt = PLAIN_PROMPT_TEMPLATE.format(question=question)
        return prompt, f"{prompt} {answer}"

    if prompt_format == "plain_mcq":
        prompt = PLAIN_MCQ_PROMPT_TEMPLATE.format(question=question)
        return prompt, f"{prompt} {answer}"

    if prompt_format == "chat_template":
        if tokenizer.chat_template is None:
            raise ValueError(
                "prompt_format=chat_template requires tokenizer.chat_template. "
                "Use an Instruct tokenizer or set prompt_format=plain_qa."
            )
        prompt_messages = [{"role": "user", "content": question}]
        full_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt, full_text

    raise ValueError(f"Unknown prompt_format: {prompt_format}")


def make_collator(tokenizer):
    def collate(batch: list[dict]) -> dict:
        input_rows = [
            {
                "input_ids": row["input_ids"],
                "attention_mask": row["attention_mask"],
            }
            for row in batch
        ]
        padded = tokenizer.pad(input_rows, padding=True, return_tensors="pt")
        max_len = padded["input_ids"].shape[1]
        labels = []
        for row in batch:
            label = row["labels"]
            labels.append(label + [-100] * (max_len - len(label)))
        padded["labels"] = torch.tensor(labels, dtype=torch.long)
        return padded

    return collate


def load_examples(data_cfg: dict, tokenizer=None, is_main: bool = True) -> list[TofuExample]:
    examples: list[TofuExample] = []
    for split_cfg in data_cfg["splits"]:
        split_examples = load_config_examples(data_cfg, split_cfg, tokenizer=tokenizer)
        examples.extend(split_examples)
        label = split_cfg.get("name", split_cfg.get("config_name", data_cfg.get("dataset_name", "dataset")))
        block_info = ""
        if data_cfg.get("dataset_format") == "muse_text":
            block_info = (
                f"; block_tokens={split_cfg.get('text_block_tokens', data_cfg.get('text_block_tokens', data_cfg.get('max_length')))}"
                f"; chunk_chars={split_cfg.get('text_chunk_chars', data_cfg.get('text_chunk_chars', 2048))}"
            )
        rank0_print(is_main, f"loaded {len(split_examples)} examples from {label}{block_info}")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    train_cfg = cfg["train"]
    is_distributed, rank, local_rank, world_size, dist_device = setup_distributed(train_cfg)
    is_main = rank == 0

    model, tokenizer = load_model_and_tokenizer(cfg["model"])
    data_cfg = cfg["data"]

    examples = load_examples(data_cfg, tokenizer=tokenizer, is_main=is_main)
    rank0_print(is_main, f"total fine-tune examples: {len(examples)}")

    dataset = TofuSFTDataset(
        examples,
        tokenizer,
        data_cfg["max_length"],
        data_cfg.get("prompt_format", "chat_template"),
    )
    sampler = None
    batch_size = train_cfg["batch_size"]
    if is_distributed:
        batch_size_mode = train_cfg.get("fsdp_batch_size_mode", "per_rank")
        if batch_size_mode == "global":
            if batch_size % world_size != 0:
                raise ValueError(
                    "For FSDP with fsdp_batch_size_mode=global, train.batch_size "
                    f"must be divisible by world_size={world_size}. Got {batch_size}."
                )
            batch_size = batch_size // world_size
            global_micro_batch = train_cfg["batch_size"]
        elif batch_size_mode == "per_rank":
            global_micro_batch = batch_size * world_size
        else:
            raise ValueError(
                "train.fsdp_batch_size_mode must be 'per_rank' or 'global'. "
                f"Got {batch_size_mode!r}."
            )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.get("seed", 42),
            drop_last=False,
        )
        rank0_print(
            is_main,
            f"FSDP enabled: world_size={world_size}, local_rank={local_rank}, "
            f"batch_size_mode={batch_size_mode}, per-rank batch_size={batch_size}, "
            f"global micro-batch={global_micro_batch}",
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=make_collator(tokenizer),
    )

    if is_distributed:
        if hasattr(model, "hf_device_map"):
            raise ValueError(
                "FSDP requires loading the model without device_map. "
                "Use the FSDP launcher, which sets model.device=cpu in a temporary config."
            )
        if train_cfg.get("fsdp_align_param_dtype", True):
            cast_floating_params(model, fsdp_param_dtype(train_cfg), is_main)
        enable_memory_saving(model, train_cfg, is_main)
        model = wrap_fsdp(model, train_cfg, dist_device, is_main=is_main)
        device = dist_device
    else:
        device = model_device(model)
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check model.use_lora or adapter config.")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    first_param = next(model.parameters())
    rank0_print(
        is_main,
        f"trainable params: {trainable_count:,} / {total_params:,} "
        f"({trainable_count / total_params:.4%})"
    )
    rank0_print(is_main, f"first parameter dtype: {first_param.dtype}, device: {first_param.device}")

    adam_eps = train_cfg.get("adam_eps", 1.0e-8)
    opt = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        eps=adam_eps,
        foreach=train_cfg.get("adam_foreach", None),
        fused=train_cfg.get("adam_fused", None),
    )
    rank0_print(
        is_main,
        f"optimizer: AdamW lr={train_cfg['lr']} weight_decay={train_cfg.get('weight_decay', 0.0)} "
        f"eps={adam_eps} foreach={train_cfg.get('adam_foreach', None)} fused={train_cfg.get('adam_fused', None)}"
    )

    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    epochs = train_cfg.get("epochs", 1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    log_every = train_cfg.get("log_every", 20)
    total_updates = math.ceil(len(loader) / grad_accum) * epochs
    update = 0

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running = 0.0
        logged_steps = 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            if epoch == 0 and step == 1:
                supervised_tokens = (batch["labels"] != -100).sum().item()
                total_tokens = batch["labels"].numel()
                rank0_print(is_main, f"first batch supervised tokens: {supervised_tokens}/{total_tokens}")
            should_update = step % grad_accum == 0 or step == len(loader)
            loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss
            if epoch == 0 and step == 1:
                rank0_print(is_main, f"initial loss before update: {loss.detach().item():.6f}")
            loss_is_finite = torch.isfinite(loss)
            if is_distributed:
                loss_finite_flag = loss_is_finite.to(device=device, dtype=torch.int32)
                dist.all_reduce(loss_finite_flag, op=dist.ReduceOp.MIN)
                loss_is_finite = loss_finite_flag.bool()
            if not bool(loss_is_finite.item()):
                supervised_tokens = (batch["labels"] != -100).sum().item()
                rank0_print(
                    is_main,
                    f"non-finite loss before backward on at least one rank; skipping batch globally: "
                    f"epoch={epoch + 1} step={step} update={update} "
                    f"loss={loss.detach().item()} supervised_tokens={supervised_tokens} "
                    f"input_len={batch['input_ids'].shape[-1]}"
                )
                if should_update:
                    opt.zero_grad(set_to_none=True)
                continue
            (loss / grad_accum).backward()
            running += float(loss.detach().item())
            logged_steps += 1

            if should_update:
                if is_distributed:
                    grad_norm = model.clip_grad_norm_(max_grad_norm)
                else:
                    grad_norm = clip_grad_norm_(trainable_params, max_grad_norm)
                grad_norm_tensor = torch.as_tensor(grad_norm, device=device)
                grad_is_finite = torch.isfinite(grad_norm_tensor)
                if is_distributed:
                    grad_finite_flag = grad_is_finite.to(device=device, dtype=torch.int32)
                    dist.all_reduce(grad_finite_flag, op=dist.ReduceOp.MIN)
                    grad_is_finite = grad_finite_flag.bool()
                if not bool(grad_is_finite.item()):
                    rank0_print(
                        is_main,
                        f"non-finite grad norm before optimizer step on at least one rank; skipping update globally: "
                        f"epoch={epoch + 1} step={step} update={update} grad_norm={grad_norm}"
                    )
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                opt.zero_grad(set_to_none=True)
                update += 1
                if update == 1 or update % log_every == 0 or update == total_updates:
                    avg = running / max(logged_steps, 1)
                    rank0_print(
                        is_main,
                        f"epoch {epoch + 1}/{epochs} update {update}/{total_updates} "
                        f"loss={loss.detach().item():.4f} avg_loss={avg:.4f}"
                    )
                    running = 0.0
                    logged_steps = 0

    output_dir = Path(cfg["output_dir"])
    model_dir = output_dir / "model"
    if is_distributed:
        save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
            state_dict = model.state_dict()
        if is_main:
            model_dir.mkdir(parents=True, exist_ok=True)
            model.module.save_pretrained(model_dir, state_dict=state_dict)
            tokenizer.save_pretrained(model_dir)
            print(f"saved fine-tuned model/adapter to {model_dir}")
        cleanup_distributed()
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"saved fine-tuned model/adapter to {model_dir}")


if __name__ == "__main__":
    main()
