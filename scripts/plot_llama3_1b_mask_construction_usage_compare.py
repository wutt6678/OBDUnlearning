from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs/llama3_1b_mask_construction_usage_compare"
PLOT_DIR = OUT_ROOT / "plots"
CONSTRUCTIONS = [
    ("saliency_grad_abs", "saliency\ngrad_abs"),
    ("saliency_weight_grad", "saliency\nweight_grad"),
    ("saliency_fisher", "saliency\nfisher"),
    ("probe_absolute", "probe\nabsolute"),
    ("probe_relative", "probe\nrelative"),
]
USAGES = [
    ("forget_only_grad_diff", "forget_only", "#2ca02c"),
    ("cadmu_grad_diff", "cadmu", "#9467bd"),
]
METRICS = [
    ("forget_rouge_l", "Forget ROUGE-L (lower better)"),
    ("utility_score", "Utility score (higher better)"),
    ("mia_auc", "MIA AUC (closer to 0.5 better)"),
    ("aggregate", "Aggregate score (higher better)"),
]

def metric(result: dict, key: str) -> float:
    if key == "forget_rouge_l":
        return result["forget"]["rouge_l"]
    if key == "utility_score":
        return result["summary"]["utility_score_higher_is_better"]
    if key == "mia_auc":
        return result["privacy"]["loss_mia_auc_forget_vs_holdout"]
    if key == "aggregate":
        return result["summary"]["aggregate_score_higher_is_better"]
    raise ValueError(key)

def load_completed() -> dict[str, dict]:
    runs = {}
    for key, _ in CONSTRUCTIONS:
        path = OUT_ROOT / key / "open_tofu_comparison.json"
        if path.exists():
            runs[key] = json.loads(path.read_text(encoding="utf-8"))
    return runs

def main() -> None:
    runs = load_completed()
    if not runs:
        raise RuntimeError(f"No completed comparisons found under {OUT_ROOT}")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5))
    axes = axes.flatten()
    x = list(range(len(CONSTRUCTIONS)))
    width = 0.22

    first_run = next(iter(runs.values()))
    original = first_run["original"]
    no_mask = first_run.get("methods", {}).get("grad_diff")

    for ax, (metric_key, title) in zip(axes, METRICS):
        for idx, (method_key, label, color) in enumerate(USAGES):
            xs = [pos + (idx - 1) * width for pos in x]
            values = []
            for construction_key, _ in CONSTRUCTIONS:
                run = runs.get(construction_key)
                if run is None or method_key not in run.get("methods", {}):
                    values.append(float("nan"))
                else:
                    values.append(metric(run["methods"][method_key], metric_key))
            ax.bar(xs, values, width=width, label=label, color=color, alpha=0.85)

        ax.axhline(metric(original, metric_key), color="#d62728", linestyle="--", linewidth=2, label="original")
        if no_mask is not None:
            ax.axhline(metric(no_mask, metric_key), color="#1f77b4", linestyle=":", linewidth=2, label="no_mask grad_diff")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in CONSTRUCTIONS])
        ax.grid(True, axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    unique = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            unique.append((handle, label))
            seen.add(label)
    fig.legend(
        [h for h, _ in unique],
        [l for _, l in unique],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("Llama3-1B mask construction and usage comparison", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out = PLOT_DIR / "mask_construction_usage_bar_forget_utility_mia_aggregate.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

if __name__ == "__main__":
    main()
