from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OBJECTIVE = "dpo"
SWEEP_ROOT = ROOT / "outputs/sweeps/llama3_1b/alpha/mask_compare_probe_relative_alpha_objective_sweeps/dpo_alpha_sweep"
PLOT_DIR = SWEEP_ROOT / "plots"

METHODS = [
    (OBJECTIVE, OBJECTIVE),
    (f"forget_only_{OBJECTIVE}", f"forget_only_{OBJECTIVE}"),
    (f"cadmu_{OBJECTIVE}", f"cadmu_{OBJECTIVE}"),
]
METRICS = [
    ("forget_rouge_l", "Forget ROUGE-L (lower better)"),
    ("utility_score", "Utility score (higher better)"),
    ("privacy", "MIA AUC (lower better)"),
]
COLORS = {
    OBJECTIVE: "#1f77b4",
    f"forget_only_{OBJECTIVE}": "#2ca02c",
    f"cadmu_{OBJECTIVE}": "#9467bd",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_x(path: Path) -> float:
    return int(path.parent.name.split("_")[-1]) / 100.0


def metric(result: dict, key: str) -> float:
    if key == "forget_rouge_l":
        return result["forget"]["rouge_l"]
    if key == "utility_score":
        return result["summary"]["utility_score_higher_is_better"]
    if key == "privacy":
        return result["privacy"]["loss_mia_auc_forget_vs_holdout"]
    raise ValueError(f"Unsupported metric: {key}")


def main() -> None:
    paths = sorted((SWEEP_ROOT / "alpha").glob("alpha_*/open_tofu_comparison.json"))
    if not paths:
        raise RuntimeError(f"No completed runs found under {SWEEP_ROOT / 'alpha'}")
    runs = [(parse_x(path), load_json(path)) for path in paths]
    runs.sort(key=lambda item: item[0])
    original = runs[0][1]["original"]

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.8), sharex=True)
    for ax, (metric_key, title) in zip(axes, METRICS):
        for method, label in METHODS:
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
                markersize=4,
                color=COLORS[method],
                label=label,
            )
        ax.axhline(metric(original, metric_key), color="#d62728", linestyle="--", linewidth=2, label="original")
        ax.set_title(title)
        ax.set_xlabel("alpha_forget")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("raw value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Llama3-1B probe-relative alpha sweep: {OBJECTIVE}", y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out_path = PLOT_DIR / "alpha_curve_forget_rouge_l_utility_score_mia_auc.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
