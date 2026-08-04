from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import set_seed

from src.config import load_config
from src.data import load_config_examples, make_loader
from src.evaluate import save_json
from src.masks import build_dual_masks, mask_stats
from src.probe import build_probe_masks, build_relative_probe_masks, compute_behavior_probe_scores, probe_threshold
from src.models import load_model_and_tokenizer
from src.muse_eval import MUSEConfig, MUSESetConfig, evaluate_muse
from src.open_tofu_eval import EvalConfig, evaluate_open_tofu
from src.wmdp_eval import WMDPConfig, evaluate_wmdp
from src.saliency import compute_gradient_saliency, saliency_threshold
from src.unlearn import BASE_OBJECTIVES, run_unlearning

SHARED_METHODS = set(BASE_OBJECTIVES) | {"no_mask"}

def shared_root_for(out_dir: Path) -> Path:
    for path in (out_dir, *out_dir.parents):
        if path.name.endswith("_sweep"):
            return path / "shared"
    return out_dir / "shared"

def resolve_baselines(unlearn_cfg: dict) -> list[str]:
    if "baselines" in unlearn_cfg:
        return list(unlearn_cfg["baselines"])

    objective = unlearn_cfg.get("objective", "grad_diff")
    wrappers = unlearn_cfg.get("wrappers", ["none", "forget_only", "cadmu"])
    methods = []
    for wrapper in wrappers:
        if wrapper in ("none", None):
            methods.append(objective)
        else:
            methods.append(f"{wrapper}_{objective}")
    return methods

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/open_tofu_phi.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.get("seed", 42))
    model, tokenizer = load_model_and_tokenizer(cfg["model"])
    data_cfg = cfg["data"]
    forget_examples = load_config_examples(
        data_cfg,
        {
            "name": data_cfg.get("forget_config_name", data_cfg["forget_split"]),
            "split": data_cfg.get("forget_train_split", data_cfg.get("split", "train")),
            "max_samples": data_cfg.get("max_forget_train_samples"),
            "dataset_format": data_cfg.get("forget_dataset_format", data_cfg.get("dataset_format", "qa")),
            "question_key": data_cfg.get("question_key", "question"),
            "answer_key": data_cfg.get("answer_key", "answer"),
            "text_key": data_cfg.get("text_key"),
            "text_prompt_fraction": data_cfg.get("text_prompt_fraction", 0.35),
            "min_answer_chars": data_cfg.get("min_answer_chars", 32),
            "text_block_tokens": data_cfg.get("text_block_tokens"),
            "text_block_overlap_tokens": data_cfg.get("text_block_overlap_tokens", 0),
            "min_answer_tokens": data_cfg.get("min_answer_tokens", 8),
            "text_chunk_chars": data_cfg.get("text_chunk_chars"),
        },
        tokenizer=tokenizer,
    )
    retain_examples = load_config_examples(
        data_cfg,
        {
            "name": data_cfg.get("retain_config_name", data_cfg["retain_split"]),
            "split": data_cfg.get("retain_train_split", data_cfg.get("split", "train")),
            "max_samples": data_cfg.get("max_retain_train_samples"),
            "dataset_format": data_cfg.get("retain_dataset_format", data_cfg.get("dataset_format", "qa")),
            "question_key": data_cfg.get("question_key", "question"),
            "answer_key": data_cfg.get("answer_key", "answer"),
            "text_key": data_cfg.get("text_key"),
            "text_prompt_fraction": data_cfg.get("text_prompt_fraction", 0.35),
            "min_answer_chars": data_cfg.get("min_answer_chars", 32),
            "text_block_tokens": data_cfg.get("text_block_tokens"),
            "text_block_overlap_tokens": data_cfg.get("text_block_overlap_tokens", 0),
            "min_answer_tokens": data_cfg.get("min_answer_tokens", 8),
            "text_chunk_chars": data_cfg.get("text_chunk_chars"),
        },
        tokenizer=tokenizer,
    )
    print(f"forget train examples: {len(forget_examples)} ({data_cfg['forget_split']})")
    print(f"retain train examples: {len(retain_examples)} ({data_cfg['retain_split']})")

    masks, mask_metadata = build_training_masks(
        model,
        tokenizer,
        forget_examples,
        retain_examples,
        cfg,
    )
    mask_cfg = cfg.get("mask", {})
    mask_metadata = {
        **mask_metadata,
        "dynamic": bool(mask_cfg.get("dynamic", False)),
        "update_every": mask_cfg.get("update_every"),
    }
    stats = mask_stats(masks)
    save_json(out_dir / "mask_stats.json", stats)
    save_json(out_dir / "mask_metadata.json", mask_metadata)
    print("mask metadata:", mask_metadata)
    print("mask stats:", stats)

    eval_fn, eval_cfg = build_eval_runner(data_cfg, cfg)

    comparison_path = out_dir / "open_tofu_comparison.json"
    shared_root = shared_root_for(out_dir)
    shared_methods = set(cfg["unlearn"].get("shared_methods", SHARED_METHODS))
    existing_comparison = None
    if comparison_path.exists():
        with comparison_path.open("r", encoding="utf-8") as fh:
            existing_comparison = json.load(fh)

    if existing_comparison is not None and "original" in existing_comparison:
        print("reusing original evaluation from existing comparison...")
        original_eval = existing_comparison["original"]
        save_json(shared_root / "original" / "open_tofu_eval.json", original_eval)
    else:
        original_eval = load_or_run_shared_eval(
            shared_root=shared_root,
            name="original",
            model=model,
            tokenizer=tokenizer,
            eval_cfg=eval_cfg,
            out_dir=out_dir / "original",
            eval_fn=eval_fn,
        )
    del model
    clear_cuda()

    baselines = resolve_baselines(cfg["unlearn"])
    comparison = existing_comparison or {
        "mask_stats": stats,
        "mask_metadata": mask_metadata,
        "original": original_eval,
        "methods": {},
    }
    comparison["mask_stats"] = stats
    comparison["mask_metadata"] = mask_metadata
    comparison["original"] = original_eval
    comparison.setdefault("methods", {})

    completed_methods = set(comparison["methods"].keys())
    for method in baselines:
        if method in completed_methods:
            continue
        method_dir = out_dir / method
        eval_path = method_dir / "open_tofu_eval.json"
        shared_eval_path = shared_root / method / "open_tofu_eval.json"
        if method in shared_methods and shared_eval_path.exists():
            print(f"reusing shared eval for completed method: {method}")
            with shared_eval_path.open("r", encoding="utf-8") as fh:
                comparison["methods"][method] = json.load(fh)
            save_json(eval_path, comparison["methods"][method])
            completed_methods.add(method)
            continue
        if eval_path.exists():
            print(f"reusing existing eval for completed method: {method}")
            with eval_path.open("r", encoding="utf-8") as fh:
                comparison["methods"][method] = json.load(fh)
            if method in shared_methods:
                save_json(shared_eval_path, comparison["methods"][method])
            completed_methods.add(method)

    missing_methods = [method for method in baselines if method not in completed_methods]
    if not missing_methods:
        print(f"all requested methods already completed in {out_dir}")
    for method in missing_methods:
        print(f"\n=== running method: {method} ===")
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.get("seed", 42))
        model, tokenizer = load_model_and_tokenizer(cfg["model"])
        unlearn_one_method(model, tokenizer, forget_examples, retain_examples, masks, cfg, method)
        method_eval = eval_fn(model, tokenizer, eval_cfg, method_dir)
        save_json(method_dir / "open_tofu_eval.json", method_eval)
        if method in shared_methods:
            save_json(shared_root / method / "open_tofu_eval.json", method_eval)
        comparison["methods"][method] = method_eval
        if cfg.get("save_models", False):
            model.save_pretrained(method_dir / "model")
            tokenizer.save_pretrained(method_dir / "model")
        del model
        clear_cuda()

    comparison["summary"] = summarize_methods(comparison)
    save_json(comparison_path, comparison)
    write_comparison_markdown(out_dir / "open_tofu_comparison.md", comparison)
    print(f"saved OpenUnlearning-style outputs to {out_dir}")

def build_eval_runner(data_cfg: dict, cfg: dict):
    benchmark = data_cfg.get("benchmark_eval", "tofu").lower()
    if benchmark == "muse":
        sets = [MUSESetConfig(**item) for item in data_cfg["muse_eval_sets"]]
        return evaluate_muse, MUSEConfig(
            sets=sets,
            max_length=data_cfg.get("eval_max_length", data_cfg.get("max_length", 512)),
            max_new_tokens=cfg.get("eval", {}).get("max_new_tokens", 64),
            text_prompt_fraction=data_cfg.get("text_prompt_fraction", 0.35),
            prompt_format=data_cfg.get("prompt_format", "plain_qa"),
        )
    if benchmark == "wmdp":
        return evaluate_wmdp, WMDPConfig(
            dataset_name=data_cfg.get("eval_dataset_name", data_cfg.get("dataset_name", "cais/wmdp")),
            subsets=tuple(data_cfg.get("wmdp_eval_subsets", ["wmdp-bio", "wmdp-chem", "wmdp-cyber"])),
            split=data_cfg.get("eval_split", "test"),
            max_samples=data_cfg.get("max_eval_samples"),
            max_length=data_cfg.get("eval_max_length", data_cfg.get("max_length", 512)),
        )
    if benchmark != "tofu":
        raise ValueError(f"Unsupported benchmark_eval for run_open_tofu.py: {benchmark}")
    return evaluate_open_tofu, EvalConfig(
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

def load_or_run_shared_eval(shared_root: Path, name: str, model, tokenizer, eval_cfg, out_dir: Path, eval_fn=evaluate_open_tofu):
    shared_eval_path = shared_root / name / "open_tofu_eval.json"
    if shared_eval_path.exists():
        print(f"reusing shared eval for {name}...")
        with shared_eval_path.open("r", encoding="utf-8") as fh:
            eval_result = json.load(fh)
        save_json(out_dir / "open_tofu_eval.json", eval_result)
        return eval_result

    print(f"evaluating {name}...")
    eval_result = eval_fn(model, tokenizer, eval_cfg, out_dir)
    save_json(out_dir / "open_tofu_eval.json", eval_result)
    save_json(shared_eval_path, eval_result)
    return eval_result

def build_training_masks(model, tokenizer, forget_examples, retain_examples, cfg: dict):
    data_cfg = cfg["data"]
    mask_cfg = cfg.get("mask", {})
    method = mask_cfg.get("method", "saliency")

    if method == "saliency":
        sal_cfg = cfg["saliency"]
        forget_loader = make_loader(
            forget_examples,
            tokenizer,
            data_cfg["max_length"],
            sal_cfg["batch_size"],
            shuffle=False,
        )
        retain_loader = make_loader(
            retain_examples,
            tokenizer,
            data_cfg["max_length"],
            sal_cfg["batch_size"],
            shuffle=False,
        )
        mode = sal_cfg.get("mode", "grad_abs")
        print(f"computing forget saliency ({mode})...")
        forget_scores = compute_gradient_saliency(model, forget_loader, sal_cfg["batches"], mode)
        print(f"computing retain saliency ({mode})...")
        retain_scores = compute_gradient_saliency(model, retain_loader, sal_cfg["batches"], mode)
        forget_threshold = saliency_threshold(forget_scores, sal_cfg["forget_quantile"])
        retain_threshold = saliency_threshold(retain_scores, sal_cfg["retain_quantile"])
        masks = build_dual_masks(retain_scores, forget_scores, retain_threshold, forget_threshold)
        return masks, {
            "method": "saliency",
            "mode": mode,
            "forget_threshold": forget_threshold,
            "retain_threshold": retain_threshold,
            "forget_quantile": sal_cfg["forget_quantile"],
            "retain_quantile": sal_cfg["retain_quantile"],
        }

    if method == "probe":
        probe_cfg = cfg.get("probe", {})
        forget_loader = make_loader(
            forget_examples,
            tokenizer,
            data_cfg["max_length"],
            probe_cfg.get("batch_size", cfg["saliency"].get("batch_size", 1)),
            shuffle=False,
        )
        retain_loader = make_loader(
            retain_examples,
            tokenizer,
            data_cfg["max_length"],
            probe_cfg.get("batch_size", cfg["saliency"].get("batch_size", 1)),
            shuffle=False,
        )
        batches = probe_cfg.get("batches", cfg["saliency"].get("batches", 1))
        module_filter = probe_cfg.get("module_filter", "trainable")
        print("computing forget behavior probe...")
        forget_scores = compute_behavior_probe_scores(model, forget_loader, batches, module_filter)
        print("computing retain behavior probe...")
        retain_scores = compute_behavior_probe_scores(model, retain_loader, batches, module_filter)
        grouping = probe_cfg.get("grouping", "absolute")
        if grouping == "absolute":
            forget_quantile = probe_cfg.get("forget_quantile", cfg["saliency"].get("forget_quantile", 0.70))
            retain_quantile = probe_cfg.get("retain_quantile", cfg["saliency"].get("retain_quantile", 0.90))
            forget_threshold = probe_threshold(forget_scores, forget_quantile)
            retain_threshold = probe_threshold(retain_scores, retain_quantile)
            masks = build_probe_masks(
                model,
                retain_scores,
                forget_scores,
                retain_threshold,
                forget_threshold,
                unmapped=probe_cfg.get("unmapped", "neutral"),
            )
            metadata = {
                "forget_threshold": forget_threshold,
                "retain_threshold": retain_threshold,
                "forget_quantile": forget_quantile,
                "retain_quantile": retain_quantile,
            }
        elif grouping == "relative":
            activity_quantile = probe_cfg.get("activity_quantile", 0.30)
            margin_quantile = probe_cfg.get("margin_quantile", 0.50)
            common_names = set(retain_scores) & set(forget_scores)
            retain_margin_scores = {
                name: retain_scores[name] - forget_scores[name]
                for name in common_names
            }
            forget_margin_scores = {
                name: forget_scores[name] - retain_scores[name]
                for name in common_names
            }
            retain_activity_threshold = probe_threshold(retain_scores, activity_quantile)
            forget_activity_threshold = probe_threshold(forget_scores, activity_quantile)
            retain_margin_threshold = probe_threshold(retain_margin_scores, margin_quantile)
            forget_margin_threshold = probe_threshold(forget_margin_scores, margin_quantile)
            masks = build_relative_probe_masks(
                model,
                retain_scores,
                forget_scores,
                retain_activity_threshold,
                forget_activity_threshold,
                retain_margin_threshold,
                forget_margin_threshold,
                unmapped=probe_cfg.get("unmapped", "neutral"),
            )
            metadata = {
                "retain_activity_threshold": retain_activity_threshold,
                "forget_activity_threshold": forget_activity_threshold,
                "retain_margin_threshold": retain_margin_threshold,
                "forget_margin_threshold": forget_margin_threshold,
                "activity_quantile": activity_quantile,
                "margin_quantile": margin_quantile,
            }
        else:
            raise ValueError(f"Unknown probe grouping: {grouping}")
        return masks, {
            "method": "probe",
            "grouping": grouping,
            "behavior": "mean_abs_activation",
            "module_filter": module_filter,
            "probed_modules": len(forget_scores),
            "unmapped": probe_cfg.get("unmapped", "neutral"),
            **metadata,
        }

    raise ValueError(f"Unknown mask method: {method}")

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
    overrides = method_cfg.pop("method_overrides", {}) or {}
    if method in overrides:
        method_cfg.update(overrides[method])
    method_cfg["method"] = method

    mask_cfg = cfg.get("mask", {})
    refresh_masks = None
    dynamic_history = []
    if mask_cfg.get("dynamic", False):
        method_cfg["dynamic_masking"] = True
        method_cfg["mask_update_every"] = int(mask_cfg.get("update_every", 0))
        dynamic_history.append(
            {
                "step": 0,
                "stats": mask_stats(masks),
                "metadata": {
                    "dynamic": True,
                    "update_every": method_cfg["mask_update_every"],
                    "initial": True,
                },
            }
        )

        def refresh_masks(current_model, step: int):
            refreshed_masks, refreshed_metadata = build_training_masks(
                current_model,
                tokenizer,
                forget_examples,
                retain_examples,
                cfg,
            )
            refreshed_stats = mask_stats(refreshed_masks)
            dynamic_history.append(
                {
                    "step": step,
                    "stats": refreshed_stats,
                    "metadata": refreshed_metadata,
                }
            )
            print("dynamic mask metadata:", refreshed_metadata)
            print("dynamic mask stats:", refreshed_stats)
            return refreshed_masks

    run_unlearning(
        model,
        train_forget_loader,
        train_retain_loader,
        masks,
        method_cfg,
        refresh_masks=refresh_masks,
    )
    if dynamic_history:
        save_json(Path(cfg["output_dir"]) / method / "dynamic_mask_history.json", dynamic_history)

def summarize_methods(comparison: dict) -> dict:
    original = comparison["original"].get("summary", {})
    rows = {}
    for method, result in comparison["methods"].items():
        summary = result.get("summary", {})
        row = dict(summary)
        for key, value in summary.items():
            if isinstance(value, (int, float)) and isinstance(original.get(key), (int, float)):
                row[f"{key}_delta"] = value - original[key]
        rows[method] = row
    return rows

def write_comparison_markdown(path: Path, comparison: dict):
    if is_tofu_summary(comparison):
        lines = [
            "# OpenUnlearning-Style TOFU Comparison",
            "",
            "| method | mem score | mem delta | utility | utility delta | privacy | aggregate | aggregate delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for method, row in comparison["summary"].items():
            lines.append(
                f"| {method} | {row['memorization_score_higher_is_better']:.4f} | "
                f"{row['memorization_score_higher_is_better_delta']:.4f} | {row['utility_score_higher_is_better']:.4f} | "
                f"{row['utility_score_higher_is_better_delta']:.4f} | {row['privacy_score_higher_is_better']:.4f} | "
                f"{row['aggregate_score_higher_is_better']:.4f} | {row['aggregate_score_higher_is_better_delta']:.4f} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    summary_rows = comparison.get("summary", {})
    metric_keys = sorted({key for row in summary_rows.values() for key in row if isinstance(row[key], (int, float))})
    lines = ["# Benchmark Comparison", ""]
    if not metric_keys:
        lines.append("No numeric summary metrics were found.")
    else:
        lines.append("| method | " + " | ".join(metric_keys) + " |")
        lines.append("|---" + "|---:" * len(metric_keys) + "|")
        for method, row in summary_rows.items():
            vals = [f"{row[key]:.4f}" if key in row else "" for key in metric_keys]
            lines.append("| " + method + " | " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def is_tofu_summary(comparison: dict) -> bool:
    original = comparison.get("original", {}).get("summary", {})
    return {
        "memorization_score_higher_is_better",
        "utility_score_higher_is_better",
        "privacy_score_higher_is_better",
        "aggregate_score_higher_is_better",
    }.issubset(original)

def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
