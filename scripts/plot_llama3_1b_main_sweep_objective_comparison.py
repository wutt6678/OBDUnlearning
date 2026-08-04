from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRAD_DIFF_PATH = (
    ROOT
    / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep"
    / "activity"
    / "activity_q_10"
    / "open_tofu_comparison.json"
)
GRAD_ASCENT_PATH = (
    ROOT
    / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep_grad_ascent_sweep"
    / "activity"
    / "activity_q_10"
    / "open_tofu_comparison.json"
)
PLOT_DIR = (
    ROOT
    / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep_grad_ascent_sweep"
    / "plots"
)

WRAPPERS = [
    ("none", "No mask"),
    ("forget_only", "Forget only"),
    ("cadmu", "CADMU"),
]
METRICS = [
    ("forget_rouge_l", "Forget ROUGE-L (lower better)"),
    ("utility_score", "Utility score (higher better)"),
    ("privacy", "MIA AUC (lower better)"),
]

def load_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Missing completed experiment: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def method_name(wrapper: str, objective: str) -> str:
    return objective if wrapper == "none" else f"{wrapper}_{objective}"

def metric(result: dict, key: str) -> float:
    if key == "forget_rouge_l":
        return result["forget"]["rouge_l"]
    if key == "utility_score":
        return result["summary"]["utility_score_higher_is_better"]
    return result["privacy"]["loss_mia_auc_forget_vs_holdout"]

def main() -> None:
    grad_diff = load_json(GRAD_DIFF_PATH)
    grad_ascent = load_json(GRAD_ASCENT_PATH)
    x = np.arange(len(WRAPPERS))
    width = 0.36

    fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.8))
    for ax, (metric_key, title) in zip(axes, METRICS):
        diff_values = [
            metric(
                grad_diff["methods"][method_name(wrapper, "grad_diff")],
                metric_key,
            )
            for wrapper, _ in WRAPPERS
        ]
        ascent_values = [
            metric(
                grad_ascent["methods"][method_name(wrapper, "grad_ascent")],
                metric_key,
            )
            for wrapper, _ in WRAPPERS
        ]

        ax.bar(
            x - width / 2,
            diff_values,
            width,
            color="#1f77b4",
            label="grad_diff",
        )
        ax.bar(
            x + width / 2,
            ascent_values,
            width,
            color="#ff7f0e",
            label="grad_ascent",
        )
        ax.axhline(
            metric(grad_diff["original"], metric_key),
            color="#d62728",
            linestyle="--",
            linewidth=2,
            label="original",
        )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in WRAPPERS], rotation=15)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("raw value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Llama3-1B main_sweep: grad_diff vs grad_ascent", y=1.02)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "grad_diff_vs_grad_ascent_forget_rouge_l_utility_score_mia_auc.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

if __name__ == "__main__":
    main()
