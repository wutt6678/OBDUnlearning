from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
METRICS = [
    ("rouge", "Forget ROUGE-L (lower better)"),
    ("utility_score", "Utility score (higher better)"),
    ("mia", "MIA AUC (closer to 0.5 better)"),
]
COLORS = {
    "grad_diff": "#1f77b4",
    "forget_only_grad_diff": "#2ca02c",
    "cadmu_grad_diff": "#9467bd",
}

def value(result: dict, metric: str) -> float:
    if metric == "rouge":
        return result["forget"]["rouge_l"]
    if metric == "utility_score":
        return result["summary"]["utility_score_higher_is_better"]
    return result["privacy"]["loss_mia_auc_forget_vs_holdout"]

def plot(root: Path, group: str, prefix: str, x_label: str, methods: list[str]) -> None:
    paths = sorted(root.glob(f"{group}/{prefix}*/open_tofu_comparison.json"))
    if not paths:
        print(f"skipping {root.name}/{group}: no completed runs")
        return
    runs = []
    for path in paths:
        x = int(path.parent.name.split("_")[-1]) / 100
        runs.append((x, json.loads(path.read_text(encoding="utf-8"))))
    runs.sort()

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True)
    original = runs[0][1]["original"]
    for ax, (metric, title) in zip(axes, METRICS):
        for method in methods:
            points = [
                (x, value(run["methods"][method], metric))
                for x, run in runs
                if method in run.get("methods", {})
            ]
            if points:
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    linewidth=2,
                    color=COLORS[method],
                    label=method,
                )
        ax.axhline(
            value(original, metric),
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
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output = plot_dir / f"{group}_curve_forget_rouge_l_utility_score_mia_auc.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output}")

def main() -> None:
    full_methods = [
        "grad_diff",
        "forget_only_grad_diff",
        "cadmu_grad_diff",
        ]
    plot(
        ROOT / "outputs/sweeps/llama3_3b/alpha/mask_compare_probe_relative_alpha_sweep",
        "alpha",
        "alpha_",
        "alpha_forget",
        full_methods,
    )
    plot(
        ROOT / "outputs/sweeps/llama3_3b/beta/mask_compare_probe_relative_beta_sweep",
        "beta",
        "beta_",
        "beta_retain",
        full_methods,
    )
    stronger = (
        ROOT
        / "outputs/sweeps/llama3_3b/main/mask_compare_probe_relative_main_sweep"
    )
    plot(stronger, "activity", "activity_q_", "activity_quantile", ["grad_diff", "cadmu_grad_diff"])
    plot(stronger, "margin", "margin_q_", "margin_quantile", ["grad_diff", "cadmu_grad_diff"])

if __name__ == "__main__":
    main()
