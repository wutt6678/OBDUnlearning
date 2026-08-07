from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SWEEP_ROOT = ROOT / "outputs/sweeps/llama3_1b/main/mask_compare_probe_relative_main_sweep"
PLOT_DIR = SWEEP_ROOT / "plots"

METHODS = [
    ("grad_diff", "grad_diff"),
    ("forget_only_grad_diff", "forget_only_grad_diff"),
    ("cadmu_grad_diff", "cadmu_grad_diff"),
]

FIGURES = [
    (
        "activity",
        sorted((SWEEP_ROOT / "activity").glob("activity_q_*/open_tofu_comparison.json")),
        "activity_quantile",
    ),
    (
        "margin",
        sorted((SWEEP_ROOT / "margin").glob("margin_q_*/open_tofu_comparison.json")),
        "margin_quantile",
    ),
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
SHARED_METHODS = {"grad_diff"}
_SHARED_CACHE: dict[str, dict] = {}

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def get_original_baseline(run: dict) -> dict[str, float]:
    original = run["original"]
    return {
        "forget_rouge_l": original["forget"]["rouge_l"],
        "utility_score": original["summary"]["utility_score_higher_is_better"],
        "privacy": original["privacy"]["loss_mia_auc_forget_vs_holdout"],
    }

def parse_x_value(path: Path) -> float:
    stem = path.parent.name
    if stem.startswith("activity_q_"):
        return int(stem.split("_")[-1]) / 100.0
    if stem.startswith("margin_q_"):
        return int(stem.split("_")[-1]) / 100.0
    raise ValueError(f"Unexpected run directory name: {stem}")

def collect_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for path in paths:
        run = load_json(path)
        runs.append(
            {
                "path": path,
                "x": parse_x_value(path),
                "run": run,
            }
        )
    runs.sort(key=lambda item: item["x"])
    return runs

def shared_method_result(method_key: str) -> dict | None:
    if method_key not in SHARED_METHODS:
        return None
    if method_key not in _SHARED_CACHE:
        path = SWEEP_ROOT / "shared" / method_key / "open_tofu_eval.json"
        if not path.exists():
            return None
        _SHARED_CACHE[method_key] = load_json(path)
    return _SHARED_CACHE[method_key]

def method_result(run: dict, method_key: str) -> dict | None:
    result = run.get("methods", {}).get(method_key)
    if result is not None:
        return result
    return shared_method_result(method_key)

def metric_series(run: dict, method_key: str, metric_key: str) -> float | None:
    result = method_result(run, method_key)
    if result is None:
        return None
    if metric_key == "forget_rouge_l":
        return result["forget"]["rouge_l"]
    if metric_key == "utility_score":
        return result["summary"]["utility_score_higher_is_better"]
    if metric_key == "privacy":
        return result["privacy"]["loss_mia_auc_forget_vs_holdout"]
    raise ValueError(f"Unsupported metric: {metric_key}")

def plot_curve(fig_kind: str, runs: list[dict], x_label: str, baseline: dict[str, float]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.8), sharex=True)
    if len(METRICS) == 1:
        axes = [axes]

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        for method_key, method_label in METHODS:
            points = [
                (item["x"], metric_series(item["run"], method_key, metric_key))
                for item in runs
            ]
            points = [(x, y) for x, y in points if y is not None]
            if not points:
                continue
            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=4,
                color=COLORS[method_key],
                label=method_label,
            )

        ax.axhline(
            baseline[metric_key],
            color="#d62728",
            linestyle="--",
            linewidth=2.0,
            label="original",
        )
        ax.set_title(metric_title)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("raw value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Llama3-1B probe-relative main_sweep sweep: {fig_kind}", y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out_path = PLOT_DIR / f"{fig_kind}_curve_forget_rouge_l_utility_score_mia_auc.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

def main() -> None:
    for fig_kind, paths, x_label in FIGURES:
        runs = collect_runs(paths)
        if not runs:
            raise RuntimeError(f"No runs found for {fig_kind}")
        baseline = get_original_baseline(runs[0]["run"])
        plot_curve(fig_kind, runs, x_label, baseline)

if __name__ == "__main__":
    main()
