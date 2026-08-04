from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    ("forget_rouge_l", "Forget ROUGE-L (lower better)"),
    ("utility_score", "Utility score (higher better)"),
    ("privacy", "MIA AUC (closer to 0.5 better)"),
]
METHODS = [
    ("grad_diff", "grad_diff"),
    ("forget_only_grad_diff", "forget_only_grad_diff"),
    ("cadmu_grad_diff", "cadmu_grad_diff"),
]
COLORS = {
    "grad_diff": "#1f77b4",
    "forget_only_grad_diff": "#2ca02c",
    "cadmu_grad_diff": "#9467bd",
}
SHARED_METHODS = {"grad_diff"}
_SHARED_CACHE: dict[tuple[Path, str], dict] = {}

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def parse_x_value(path: Path) -> float:
    stem = path.parent.name
    for prefix in ("activity_q_", "margin_q_", "alpha_", "beta_"):
        if stem.startswith(prefix):
            return int(stem.removeprefix(prefix)) / 100.0
    raise ValueError(f"Unexpected run directory name: {stem}")

def metric(result: dict, key: str) -> float:
    if key == "forget_rouge_l":
        return result["forget"]["rouge_l"]
    if key == "utility_score":
        return result["summary"]["utility_score_higher_is_better"]
    if key == "privacy":
        return result["privacy"]["loss_mia_auc_forget_vs_holdout"]
    raise ValueError(f"Unsupported metric: {key}")

def shared_method_result(root: Path, method_key: str) -> dict | None:
    if method_key not in SHARED_METHODS:
        return None
    cache_key = (root, method_key)
    if cache_key not in _SHARED_CACHE:
        path = root / "shared" / method_key / "open_tofu_eval.json"
        if not path.exists():
            return None
        _SHARED_CACHE[cache_key] = load_json(path)
    return _SHARED_CACHE[cache_key]

def method_result(root: Path, run: dict, method_key: str) -> dict | None:
    result = run.get("methods", {}).get(method_key)
    if result is not None:
        return result
    return shared_method_result(root, method_key)

def collect_runs(root: Path, group: str, prefix: str) -> list[dict]:
    paths = sorted(root.glob(f"{group}/{prefix}*/open_tofu_comparison.json"))
    runs = []
    for path in paths:
        runs.append({"x": parse_x_value(path), "run": load_json(path), "path": path})
    runs.sort(key=lambda item: item["x"])
    return runs

def plot(root: Path, group: str, prefix: str, x_label: str, title_prefix: str, methods: list[tuple[str, str]] | None = None) -> None:
    methods = methods or METHODS
    runs = collect_runs(root, group, prefix)
    if not runs:
        print(f"skipping {root.name}/{group}: no completed runs")
        return

    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.8), sharex=True)
    if len(METRICS) == 1:
        axes = [axes]
    original = runs[0]["run"]["original"]

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        for method_key, method_label in methods:
            points = []
            for item in runs:
                result = method_result(root, item["run"], method_key)
                if result is not None:
                    points.append((item["x"], metric(result, metric_key)))
            if not points:
                continue
            ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                marker="o",
                linewidth=2.0,
                markersize=4,
                color=COLORS[method_key],
                label=method_label,
            )
        ax.axhline(
            metric(original, metric_key),
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
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(f"{title_prefix}: {group}", y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    output = plot_dir / f"{group}_curve_forget_rouge_l_utility_score_mia_auc.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output}")
