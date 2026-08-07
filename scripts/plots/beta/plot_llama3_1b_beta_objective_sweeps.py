from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SWEEP_ROOT = ROOT / "outputs/sweeps/llama3_1b/beta/mask_compare_probe_relative_beta_objective_sweeps"
PLOT_ROOT = SWEEP_ROOT / "plots"
OBJECTIVES = [
    "grad_diff",
    "grad_ascent",
    "npo",
    "dpo",
    "simnpo",
    "rmu",
    "kl",
    "retrain",
]
WRAPPERS = [
    ("none", "no_mask"),
    ("forget_only", "single_mask"),
    ("cadmu", "dual_mask"),
]
METRICS = [
    ("forget_rouge_l", "Forget ROUGE-L (lower better)"),
    ("utility_score", "Utility score (higher better)"),
    ("privacy", "MIA AUC (lower better)"),
]
COLORS = {
    "grad_diff": "#1f77b4",
    "grad_ascent": "#ff7f0e",
    "npo": "#2ca02c",
    "dpo": "#d62728",
    "simnpo": "#9467bd",
    "rmu": "#8c564b",
    "kl": "#e377c2",
    "retrain": "#7f7f7f",
}


def method_name(objective: str, wrapper: str) -> str:
    return objective if wrapper == "none" else f"{wrapper}_{objective}"


def metric(result: dict, key: str) -> float:
    if key == "forget_rouge_l":
        return result["forget"]["rouge_l"]
    if key == "utility_score":
        return result["summary"]["utility_score_higher_is_better"]
    if key == "privacy":
        return result["privacy"]["loss_mia_auc_forget_vs_holdout"]
    raise ValueError(f"Unsupported metric: {key}")


def load_runs(objective: str) -> list[tuple[float, dict]]:
    root = SWEEP_ROOT / f"{objective}_beta_sweep"
    paths = sorted((root / "beta").glob("beta_*/open_tofu_comparison.json"))
    runs = []
    for path in paths:
        x = int(path.parent.name.split("_")[-1]) / 100.0
        with path.open("r", encoding="utf-8") as fh:
            runs.append((x, json.load(fh)))
    runs.sort(key=lambda item: item[0])
    return runs


def plot_wrapper(wrapper: str, wrapper_label: str) -> None:
    loaded = {objective: load_runs(objective) for objective in OBJECTIVES}
    loaded = {objective: runs for objective, runs in loaded.items() if runs}
    if not loaded:
        raise RuntimeError(f"No completed beta objective sweep runs found under {SWEEP_ROOT}")
    original = next(iter(loaded.values()))[0][1]["original"]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(19, 5.2), sharex=True)
    for ax, (metric_key, title) in zip(axes, METRICS):
        for objective, runs in loaded.items():
            method = method_name(objective, wrapper)
            points = [
                (x, metric(run["methods"][method], metric_key))
                for x, run in runs
                if method in run.get("methods", {})
            ]
            if not points:
                continue
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=2,
                markersize=3.5,
                color=COLORS[objective],
                label=objective,
            )
        ax.axhline(
            metric(original, metric_key),
            color="#222222",
            linestyle="--",
            linewidth=1.8,
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
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(f"Llama3-1B beta sweep across objectives: {wrapper_label}", y=1.02)
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))

    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_ROOT / f"beta_objective_comparison_{wrapper_label}_forget_rouge_l_utility_score_mia_auc.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    for wrapper, wrapper_label in WRAPPERS:
        plot_wrapper(wrapper, wrapper_label)


if __name__ == "__main__":
    main()
