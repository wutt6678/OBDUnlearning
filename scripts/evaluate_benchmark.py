from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import set_seed

from src.config import load_config
from src.models import load_model_and_tokenizer
from src.muse_eval import MUSEConfig, MUSESetConfig, evaluate_muse
from src.wmdp_eval import WMDPConfig, evaluate_wmdp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    model, tokenizer = load_model_and_tokenizer(cfg["model"])
    benchmark = cfg["benchmark"].lower()

    if benchmark == "wmdp":
        data_cfg = cfg["data"]
        result = evaluate_wmdp(
            model,
            tokenizer,
            WMDPConfig(
                dataset_name=data_cfg.get("dataset_name", "cais/wmdp"),
                subsets=tuple(data_cfg.get("subsets", ["wmdp-bio", "wmdp-chem", "wmdp-cyber"])),
                split=data_cfg.get("split", "test"),
                max_samples=data_cfg.get("max_eval_samples"),
                max_length=data_cfg.get("max_length", 512),
            ),
            cfg["output_dir"],
        )
        print("summary:", result["summary"])
        return

    if benchmark == "muse":
        data_cfg = cfg["data"]
        sets = [MUSESetConfig(**item) for item in data_cfg["sets"]]
        result = evaluate_muse(
            model,
            tokenizer,
            MUSEConfig(
                sets=sets,
                max_length=data_cfg.get("max_length", 512),
                max_new_tokens=cfg.get("eval", {}).get("max_new_tokens", 64),
                text_prompt_fraction=data_cfg.get("text_prompt_fraction", 0.35),
                prompt_format=data_cfg.get("prompt_format", "plain_qa"),
            ),
            cfg["output_dir"],
        )
        print("summary:", result["summary"])
        return

    raise ValueError(f"Unsupported benchmark: {benchmark}")


if __name__ == "__main__":
    main()
