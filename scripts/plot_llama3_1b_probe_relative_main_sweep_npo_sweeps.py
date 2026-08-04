from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = (
    ROOT
    / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep_npo_sweep"
)
PLOT_DIR = SWEEP_ROOT / "plots"

METHODS = [
    ("npo", "npo"),
    ("forget_only_npo", "forget_only_npo"),
    ("cadmu_npo", "cadmu_npo"),
]
FIGURES = [
    ("activity", "activity_q_*", "activity_quantile"),
    ("margin", "margin_q_*", "margin_quantile"),
]
METRICS = [
    ("forget_rouge_l", "Forget ROUGE-L (lower better)"),
    ("utility_score", "Utility score (higher better)"),
    ("privacy", "MIA AUC (lower better)"),
]
COLORS = {
    "npo": "#1f77b4",
    "forget_only_npo": "#2ca02c",
    "cadmu_npo": "#9467bd",
}

def metric(result: dict, key: str) -> float:
    if key == "forget_rouge_l":
        return result["forget"]["rouge_l"]
    if key == "utility_score":
        return result["summary"]["utility_score_higher_is_better"]
    return result["privacy"]["loss_mia_auc_forget_vs_holdout"]

def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for group, pattern, x_label in FIGURES:
        paths = sorted(
            (SWEEP_ROOT / group).glob(f"{pattern}/open_tofu_comparison.json")
        )
        if not paths:
            print(f"skipping {group}: no completed runs")
            continue

        runs = []
        for path in paths:
            x = int(path.parent.name.split("_")[-1]) / 100.0
            with path.open("r", encoding="utf-8") as fh:
                runs.append((x, json.load(fh)))
        runs.sort(key=lambda item: item[0])

        fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.8), sharex=True)
        original = runs[0][1]["original"]
        for ax, (metric_key, title) in zip(axes, METRICS):
            for method, label in METHODS:
                points = [
                    (x, metric(run["methods"][method], metric_key))
                    for x, run in runs
                    if method in run.get("methods", {})
                ]
                if points:
                    ax.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        linewidth=2,
                        markersize=4,
                        color=COLORS[method],
                        label=label,
                    )
            ax.axhline(
                metric(original, metric_key),
                color="#d62728",
                linestyle="--",
                linewidth=2,
                label="original",
            )
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.grid(True, alpha=0.25)

        axes[0].set_ylabel("raw value")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.suptitle(
            f"Llama3-1B probe-relative main_sweep npo: {group}",
            y=1.02,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        out_path = (
            PLOT_DIR
            / f"{group}_curve_forget_rouge_l_utility_score_mia_auc.png"
        )
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")

if __name__ == "__main__":
    main()
