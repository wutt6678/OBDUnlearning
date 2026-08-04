from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import set_seed

from src.config import load_config
from src.models import load_model_and_tokenizer
from src.open_tofu_eval import EvalConfig, evaluate_open_tofu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    model, tokenizer = load_model_and_tokenizer(cfg["model"])
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})

    result = evaluate_open_tofu(
        model,
        tokenizer,
        EvalConfig(
            dataset_name=data_cfg["dataset_name"],
            forget_perturbed_split=data_cfg.get("forget_perturbed_split", "forget10_perturbed"),
            retain_perturbed_split=data_cfg.get("retain_perturbed_split", "retain_perturbed"),
            holdout_split=data_cfg.get("holdout_split", "holdout10"),
            real_authors_split=data_cfg.get("real_authors_split", "real_authors"),
            world_facts_split=data_cfg.get("world_facts_split", "world_facts"),
            max_samples=data_cfg.get("max_eval_samples"),
            batch_size=eval_cfg.get("batch_size", 8),
            max_length=data_cfg["max_length"],
            max_new_tokens=eval_cfg.get("max_new_tokens", data_cfg.get("generation_max_new_tokens", 64)),
        ),
        cfg["output_dir"],
    )
    print("summary:", result["summary"])
    print(f"saved OpenTOFU eval to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
