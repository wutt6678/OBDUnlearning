from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
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


def result_path(run_dir: Path) -> Path:
    comparison = run_dir / "open_tofu_comparison.json"
    if comparison.exists():
        return comparison
    raise FileNotFoundError(comparison)


def benchmark_kind(result: dict) -> str:
    if "forget_knowledge" in result or "retain_knowledge" in result:
        return "muse"
    if "wmdp-bio" in result or "wmdp-cyber" in result:
        return "wmdp"
    if "forget" in result and "retain" in result:
        return "tofu"
    raise ValueError(f"Could not infer benchmark kind from keys: {sorted(result.keys())}")


def metrics_for(kind: str) -> list[tuple[str, str]]:
    if kind == "muse":
        return [
            ("forget_loss", "Forget QA loss (lower better)"),
            ("forget_rouge_l", "Forget QA ROUGE-L (lower after unlearning)"),
            ("retain_probability", "Retain QA probability (higher better)"),
            ("mean_exact", "Mean exact containment (higher before unlearning)"),
        ]
    if kind == "wmdp":
        return [
            ("summary_accuracy", "WMDP summary accuracy (higher better)"),
            ("forget_bio_accuracy", "Forget subset bio accuracy (lower after unlearning)"),
            ("retain_cyber_accuracy", "Retain subset cyber accuracy (higher better)"),
            ("summary_gold_prob", "WMDP summary gold probability (higher better)"),
        ]
    if kind == "tofu":
        return [
            ("forget_rouge_l", "Forget ROUGE-L (lower better)"),
            ("utility_score", "Utility score (higher better)"),
            ("privacy", "MIA AUC (closer to 0.5 better)"),
        ]
    raise ValueError(f"Unsupported benchmark kind: {kind}")


def metric(result: dict, key: str) -> float:
    if key == "forget_loss":
        return result["forget_knowledge"]["answer_loss_lower_is_better"]
    if key == "forget_rouge_l":
        if "forget_knowledge" in result:
            return result["forget_knowledge"]["rouge_l_higher_is_better"]
        return result["forget"]["rouge_l"]
    if key == "retain_probability":
        return result["retain_knowledge"]["answer_probability_higher_is_better"]
    if key == "mean_exact":
        return result["summary"]["mean_exact_containment_higher_is_better"]
    if key == "summary_accuracy":
        return result["summary"]["accuracy_higher_is_better"]
    if key == "forget_bio_accuracy":
        return result["wmdp-bio"]["accuracy_higher_is_better"]
    if key == "retain_cyber_accuracy":
        return result["wmdp-cyber"]["accuracy_higher_is_better"]
    if key == "summary_gold_prob":
        return result["summary"]["gold_choice_probability_higher_is_better"]
    if key == "utility_score":
        return result["summary"]["utility_score_higher_is_better"]
    if key == "privacy":
        return result["privacy"]["loss_mia_auc_forget_vs_holdout"]
    raise ValueError(f"Unsupported metric: {key}")


def shared_method_result(root: Path, method_key: str) -> dict | None:
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
    method_path = run.get("_path")
    if method_path is not None:
        candidate = Path(method_path).parent / method_key / "open_tofu_eval.json"
        if candidate.exists():
            return load_json(candidate)
    return shared_method_result(root, method_key)


def collect_runs(root: Path, group: str, prefix: str) -> list[dict]:
    paths = sorted(root.glob(f"{group}/{prefix}*/open_tofu_comparison.json"))
    runs = []
    for path in paths:
        run = load_json(path)
        run["_path"] = str(path)
        runs.append({"x": parse_x_value(path), "run": run, "path": path})
    runs.sort(key=lambda item: item["x"])
    return runs


def infer_kind(runs: list[dict], requested: str) -> str:
    if requested != "auto":
        return requested
    for item in runs:
        original = item["run"].get("original")
        if original:
            return benchmark_kind(original)
        for value in item["run"].get("methods", {}).values():
            return benchmark_kind(value)
    raise ValueError("Cannot infer benchmark kind from empty runs")


def plot(
    root: Path,
    group: str,
    prefix: str,
    x_label: str,
    title_prefix: str,
    benchmark: str = "auto",
    methods: list[tuple[str, str]] | None = None,
) -> None:
    methods = methods or METHODS
    runs = collect_runs(root, group, prefix)
    if not runs:
        print(f"skipping {root.name}/{group}: no completed runs")
        return

    kind = infer_kind(runs, benchmark)
    metrics = metrics_for(kind)
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.8), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    original = runs[0]["run"].get("original")

    for ax, (metric_key, metric_title) in zip(axes, metrics):
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
                color=COLORS.get(method_key),
                label=method_label,
            )
        if original is not None:
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
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(5, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.suptitle(f"{title_prefix}: {group}", y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    metric_slug = "_".join(key for key, _ in metrics)
    output = plot_dir / f"{group}_curve_{metric_slug}.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--benchmark", default="auto", choices=["auto", "muse", "wmdp", "tofu"])
    parser.add_argument("--title", default=None)
    parser.add_argument("--groups", nargs="*", default=None)
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    title = args.title or root.name
    groups = args.groups or ["alpha", "beta", "activity", "margin"]
    specs = {
        "alpha": ("alpha_", "alpha_forget"),
        "beta": ("beta_", "beta_retain"),
        "activity": ("activity_q_", "activity_quantile"),
        "margin": ("margin_q_", "margin_quantile"),
    }
    for group in groups:
        if group not in specs:
            raise ValueError(f"Unsupported group: {group}")
        prefix, x_label = specs[group]
        plot(root, group, prefix, x_label, title, benchmark=args.benchmark)


if __name__ == "__main__":
    main()
