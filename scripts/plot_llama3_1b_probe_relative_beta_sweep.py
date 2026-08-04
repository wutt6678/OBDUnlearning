from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "outputs/llama3_1b_mask_compare_probe_relative_beta_objective_sweeps/grad_diff_beta_sweep"
PLOT_DIR = SWEEP_ROOT / "plots"

METHODS = [
    ("grad_diff", "grad_diff"),
    ("forget_only_grad_diff", "forget_only_grad_diff"),
    ("cadmu_grad_diff", "cadmu_grad_diff"),
]
METRICS = [
    ("forget_rouge_l", "Forget ROUGE-L (lower better)"),
    ("utility_score", "Utility score (higher better)"),
    ("privacy", "MIA AUC (lower better)"),
]
COLORS = {
    "grad_diff": "#1f77b4",
    "forget_only_grad_diff": "#2ca02c",
    "cadmu_grad_diff": "#9467bd",
}

def metric(result: dict, key: str) -> float:
    if key == "forget_rouge_l":
        return result["forget"]["rouge_l"]
    if key == "utility_score":
        return result["summary"]["utility_score_higher_is_better"]
    return result["privacy"]["loss_mia_auc_forget_vs_holdout"]

def main() -> None:
    paths = sorted(
        (SWEEP_ROOT / "beta").glob("beta_*/open_tofu_comparison.json")
    )
    if not paths:
        raise RuntimeError(f"No completed beta sweep runs found under {SWEEP_ROOT}")

    runs = []
    for path in paths:
        beta = int(path.parent.name.removeprefix("beta_")) / 100.0
        with path.open("r", encoding="utf-8") as fh:
            runs.append((beta, json.load(fh)))
    runs.sort(key=lambda item: item[0])

    expected = {method for method, _ in METHODS}
    for beta, run in runs:
        missing = expected - set(run.get("methods", {}))
        if missing:
            raise RuntimeError(
                f"beta={beta:g} is missing methods: {', '.join(sorted(missing))}"
            )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.8), sharex=True)
    xs = [beta for beta, _ in runs]
    original = runs[0][1]["original"]

    for ax, (metric_key, title) in zip(axes, METRICS):
        for method_key, label in METHODS:
            ys = [
                metric(run["methods"][method_key], metric_key)
                for _, run in runs
            ]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=4,
                color=COLORS[method_key],
                label=label,
            )
        ax.axhline(
            metric(original, metric_key),
            color="#d62728",
            linestyle="--",
            linewidth=2.0,
            label="original",
        )
        ax.set_title(title)
        ax.set_xlabel("beta_retain")
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
    fig.suptitle("Llama3-1B probe-relative beta_retain sweep", y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out_path = PLOT_DIR / "beta_curve_forget_rouge_l_utility_score_mia_auc.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

if __name__ == "__main__":
    main()
