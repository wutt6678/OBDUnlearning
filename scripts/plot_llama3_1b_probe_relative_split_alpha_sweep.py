from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "outputs/llama3_1b_mask_compare_probe_relative_split_alpha_sweep"
PLOT_DIR = SWEEP_ROOT / "plots"

SPLITS = [
    ("forget01", "forget01 / retain99", SWEEP_ROOT / "forget01_alpha_sweep" / "alpha"),
    ("forget05", "forget05 / retain95", SWEEP_ROOT / "forget05_alpha_sweep" / "alpha"),
]

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

LINESTYLES = {
    "forget01": "-",
    "forget05": "--",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_x_value(path: Path) -> float:
    stem = path.parent.name
    if stem.startswith("alpha_"):
        return int(stem.split("_")[-1]) / 100.0
    raise ValueError(f"Unexpected run directory name: {stem}")


def collect_runs(run_root: Path) -> list[dict]:
    runs = []
    for path in sorted(run_root.glob("alpha_*/open_tofu_comparison.json")):
        runs.append({"path": path, "x": parse_x_value(path), "run": load_json(path)})
    runs.sort(key=lambda item: item["x"])
    return runs


def get_original_baseline(run: dict) -> dict[str, float]:
    original = run["original"]
    return {
        "forget_rouge_l": original["forget"]["rouge_l"],
        "utility_score": original["summary"]["utility_score_higher_is_better"],
        "privacy": original["privacy"]["loss_mia_auc_forget_vs_holdout"],
    }


def metric_value(run: dict, method_key: str, metric_key: str) -> float:
    if metric_key == "forget_rouge_l":
        return run["methods"][method_key]["forget"]["rouge_l"]
    if metric_key == "utility_score":
        return run["methods"][method_key]["summary"]["utility_score_higher_is_better"]
    if metric_key == "privacy":
        return run["methods"][method_key]["privacy"]["loss_mia_auc_forget_vs_holdout"]
    raise ValueError(f"Unsupported metric: {metric_key}")


def plot_split_curves(split_key: str, split_label: str, runs: list[dict]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.8), sharex=True)
    baseline = get_original_baseline(runs[0]["run"])
    xs = [item["x"] for item in runs]

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        for method_key, method_label in METHODS:
            ys = [metric_value(item["run"], method_key, metric_key) for item in runs]
            ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4, color=COLORS[method_key], label=method_label)
        ax.axhline(baseline[metric_key], color="#d62728", linestyle=":", linewidth=2.0, label="original")
        ax.set_title(metric_title)
        ax.set_xlabel("alpha_forget")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("raw value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Llama3-1B probe-relative alpha sweep: {split_label}", y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out_path = PLOT_DIR / f"{split_key}_alpha_curve_forget_rouge_l_utility_score_mia_auc.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_split_comparison(all_runs: dict[str, tuple[str, list[dict]]]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 5.2), sharex=True)

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        for split_key, (split_label, runs) in all_runs.items():
            xs = [item["x"] for item in runs]
            for method_key, method_label in METHODS:
                ys = [metric_value(item["run"], method_key, metric_key) for item in runs]
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=2.0,
                    markersize=3.5,
                    color=COLORS[method_key],
                    linestyle=LINESTYLES[split_key],
                    label=f"{split_label} {method_label}",
                )
        ax.set_title(metric_title)
        ax.set_xlabel("alpha_forget")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("raw value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Llama3-1B probe-relative alpha sweep by forget/retain split", y=1.02)
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    out_path = PLOT_DIR / "split_alpha_comparison_forget_rouge_l_utility_score_mia_auc.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    all_runs: dict[str, tuple[str, list[dict]]] = {}
    missing = []
    for split_key, split_label, run_root in SPLITS:
        runs = collect_runs(run_root)
        if not runs:
            missing.append(str(run_root))
            continue
        all_runs[split_key] = (split_label, runs)
        plot_split_curves(split_key, split_label, runs)

    if missing:
        raise RuntimeError("No completed runs found under: " + ", ".join(missing))
    plot_split_comparison(all_runs)


if __name__ == "__main__":
    main()
