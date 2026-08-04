from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locuslab/TOFU")
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--include-official-tofu-model", action="store_true")
    parser.add_argument("--all-model-files", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(args.dataset)
    print(ds)
    model_patterns = None if args.all_model_files else [
        "*.json",
        "*.txt",
        "*.model",
        "*.safetensors",
        "pytorch_model.bin",
        "merges.txt",
        "vocab.json",
        "tokenizer.*",
        "special_tokens_map.json",
    ]
    tokenizer_patterns = None if args.all_model_files else [
        "*.json",
        "*.txt",
        "*.model",
        "merges.txt",
        "vocab.json",
        "tokenizer.*",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    model_path = snapshot_download(args.model, allow_patterns=model_patterns)
    print(f"Downloaded model to: {model_path}")
    if args.tokenizer:
        tokenizer_path = snapshot_download(args.tokenizer, allow_patterns=tokenizer_patterns)
        print(f"Downloaded tokenizer to: {tokenizer_path}")
    if args.include_official_tofu_model:
        official_path = snapshot_download(
            "locuslab/tofu_ft_phi-1.5",
            allow_patterns=model_patterns,
        )
        print(f"Downloaded official TOFU Phi-1.5 checkpoint to: {official_path}")

if __name__ == "__main__":
    main()
