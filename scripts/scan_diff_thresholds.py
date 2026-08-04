from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_tofu_config_examples, make_loader
from src.evaluate import save_json
from src.masks import build_dual_masks, mask_stats
from src.models import load_model_and_tokenizer
from src.open_tofu_eval import EvalConfig, evaluate_open_tofu
from src.saliency import compute_gradient_saliency, saliency_threshold
from src.unlearn import run_unlearning

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/open_tofu_phi.yaml")
    parser.add_argument("--method", default="cadmu_grad_diff")
    parser.add_argument("--retain-quantiles", nargs="+", type=float, default=[0.85, 0.90])
    parser.add_argument("--forget-quantiles", nargs="+", type=float, default=[0.80, 0.85])
    parser.add_argument("--output-suffix", default="threshold_scan_diff")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    out_dir = Path(cfg["output_dir"]) / args.output_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    forget_examples = load_tofu_config_examples(
        data_cfg["dataset_name"],
        data_cfg["forget_split"],
        data_cfg.get("split", "train"),
        data_cfg.get("max_forget_train_samples"),
    )
    retain_examples = load_tofu_config_examples(
        data_cfg["dataset_name"],
        data_cfg["retain_split"],
        data_cfg.get("split", "train"),
        data_cfg.get("max_retain_train_samples"),
    )
    print(f"forget train examples: {len(forget_examples)} ({data_cfg['forget_split']})")
    print(f"retain train examples: {len(retain_examples)} ({data_cfg['retain_split']})")

    model, tokenizer = load_model_and_tokenizer(cfg["model"])
    sal_cfg = cfg["saliency"]
    sal_mode = sal_cfg.get("mode", "fisher")
    forget_sal_loader = make_loader(
        forget_examples,
        tokenizer,
        data_cfg["max_length"],
        sal_cfg["batch_size"],
        shuffle=False,
    )
    retain_sal_loader = make_loader(
        retain_examples,
        tokenizer,
        data_cfg["max_length"],
        sal_cfg["batch_size"],
        shuffle=False,
    )

    print(f"computing forget saliency ({sal_mode})...")
    forget_scores = compute_gradient_saliency(
        model,
        forget_sal_loader,
        sal_cfg["batches"],
        sal_mode,
    )
    print(f"computing retain saliency ({sal_mode})...")
    retain_scores = compute_gradient_saliency(
        model,
        retain_sal_loader,
        sal_cfg["batches"],
        sal_mode,
    )
    del model
    clear_cuda()

    eval_cfg = EvalConfig(
        dataset_name=data_cfg["dataset_name"],
        forget_perturbed_split=data_cfg["forget_perturbed_split"],
        retain_perturbed_split=data_cfg["retain_perturbed_split"],
        holdout_split=data_cfg["holdout_split"],
        real_authors_split=data_cfg["real_authors_split"],
        world_facts_split=data_cfg["world_facts_split"],
        max_samples=data_cfg.get("max_eval_samples"),
        batch_size=cfg["eval"]["batch_size"],
        max_length=data_cfg["max_length"],
        max_new_tokens=cfg["eval"]["max_new_tokens"],
    )

    results = {
        "method": args.method,
        "saliency_mode": sal_mode,
        "runs": {},
    }
    for retain_q in args.retain_quantiles:
        for forget_q in args.forget_quantiles:
            run_name = f"r{int(retain_q * 100):02d}_f{int(forget_q * 100):02d}"
            run_dir = out_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            retain_threshold = saliency_threshold(retain_scores, retain_q)
            forget_threshold = saliency_threshold(forget_scores, forget_q)
            masks = build_dual_masks(retain_scores, forget_scores, retain_threshold, forget_threshold)
            stats = mask_stats(masks)
            save_json(run_dir / "mask_stats.json", stats)
            print(f"\n=== {run_name}: retain_q={retain_q:.2f}, forget_q={forget_q:.2f} ===")
            print("mask stats:", json.dumps(stats, indent=2))

            set_seed(cfg.get("seed", 42))
            model, tokenizer = load_model_and_tokenizer(cfg["model"])
            unlearn_one_method(model, tokenizer, forget_examples, retain_examples, masks, cfg, args.method)
            evaluation = evaluate_open_tofu(model, tokenizer, eval_cfg, run_dir)
            results["runs"][run_name] = {
                "retain_quantile": retain_q,
                "forget_quantile": forget_q,
                "mask_stats": stats,
                "eval": evaluation,
            }
            save_json(out_dir / "scan_results.json", results)
            del model
            clear_cuda()

    write_summary(out_dir / "scan_summary.md", results)
    print(f"saved threshold scan to {out_dir}")

def unlearn_one_method(model, tokenizer, forget_examples, retain_examples, masks, cfg: dict, method: str):
    data_cfg = cfg["data"]
    train_forget_loader = make_loader(
        forget_examples,
        tokenizer,
        data_cfg["max_length"],
        cfg["unlearn"]["batch_size"],
        shuffle=True,
    )
    train_retain_loader = make_loader(
        retain_examples,
        tokenizer,
        data_cfg["max_length"],
        cfg["unlearn"]["batch_size"],
        shuffle=True,
    )
    method_cfg = dict(cfg["unlearn"])
    method_cfg["method"] = method
    run_unlearning(model, train_forget_loader, train_retain_loader, masks, method_cfg)

def write_summary(path: Path, results: dict):
    lines = [
        "# Diff Threshold Scan",
        "",
        "| run | retain q | forget q | retain only | forget only | conflict | neutral | mem | utility | privacy | aggregate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_name, row in results["runs"].items():
        stats = row["mask_stats"]
        summary = row["eval"]["summary"]
        lines.append(
            f"| {run_name} | {row['retain_quantile']:.2f} | {row['forget_quantile']:.2f} | "
            f"{stats['retain_only']:.4f} | {stats['forget_only']:.4f} | "
            f"{stats['conflict']:.4f} | {stats['neutral']:.4f} | "
            f"{summary['memorization_score_higher_is_better']:.4f} | "
            f"{summary['utility_score_higher_is_better']:.4f} | "
            f"{summary['privacy_score_higher_is_better']:.4f} | "
            f"{summary['aggregate_score_higher_is_better']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
