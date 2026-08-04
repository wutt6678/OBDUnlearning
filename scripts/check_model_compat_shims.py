from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import install_transformers_compat_shims, normalize_tied_weights_keys


def assert_mapping(value, label: str) -> None:
    if not hasattr(value, "keys"):
        raise AssertionError(f"{label}: expected mapping-like tied keys, got {type(value).__name__}")


def check_remote_style_list() -> None:
    class RemoteStyleModel:
        all_tied_weights_keys = ["lm_head.weight", "transformer.embedding.word_embeddings.weight"]

    model = RemoteStyleModel()
    normalize_tied_weights_keys(model)
    assert_mapping(model.all_tied_weights_keys, "remote-style list")


def check_native_tiny_models() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    tiny_llama = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    )
    assert_mapping(tiny_llama.all_tied_weights_keys, "tiny llama")

    tiny_qwen2 = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    )
    # Regression for the previous failure: Qwen2 must be able to assign this attr during post_init.
    tiny_qwen2.all_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    assert_mapping(tiny_qwen2.all_tied_weights_keys, "tiny qwen2")

    try:
        from transformers.models.phi3 import Phi3Config, Phi3ForCausalLM
    except Exception:
        return
    tiny_phi3 = Phi3ForCausalLM(
        Phi3Config(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    assert_mapping(tiny_phi3.all_tied_weights_keys, "tiny phi3")


def main() -> None:
    install_transformers_compat_shims()
    check_remote_style_list()
    check_native_tiny_models()
    print("model compat shims OK")


if __name__ == "__main__":
    main()
