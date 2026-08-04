from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "outputs/llm_model_unlearning_method_default_sweep"
PLOT_DIR = SWEEP_ROOT / "plots"
OBJECTIVES = ["grad_diff", "grad_ascent", "npo", "dpo", "simnpo", "rmu", "kl", "retrain"]
WRAPPERS = [
    ("none", "No mask"),
    ("forget_only", "Single mask"),
    ("cadmu", "Dual mask"),
]
METRICS = [
    ("forget_rouge_l", "Forget ROUGE-L", "lower"),
    ("utility", "Utility", "higher"),
    ("mia_auc", "MIA AUC", "closer to 0.5"),
]
COLORS = {
    "none": "#1f77b4",
    "forget_only": "#2ca02c",
    "cadmu": "#9467bd",
}


def method_name(objective: str, wrapper: str) -> str:
    return objective if wrapper == "none" else f"{wrapper}_{objective}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(method_result: dict, metric: str) -> float | None:
    try:
        if metric == "forget_rouge_l":
            return float(method_result["forget"]["rouge_l"])
        if metric == "utility":
            return float(method_result["summary"]["utility_score_higher_is_better"])
        if metric == "mia_auc":
            return float(method_result["privacy"]["loss_mia_auc_forget_vs_holdout"])
    except KeyError:
        return None
    raise ValueError(f"Unsupported metric: {metric}")


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for model_dir in sorted(SWEEP_ROOT.glob("*_sweep")):
        model = model_dir.name.removesuffix("_sweep")
        for objective in OBJECTIVES:
            comparison_path = model_dir / objective / "open_tofu_comparison.json"
            if not comparison_path.exists():
                continue
            comparison = load_json(comparison_path)
            methods = comparison.get("methods", {})
            for wrapper, wrapper_label in WRAPPERS:
                method = method_name(objective, wrapper)
                result = methods.get(method)
                if result is None:
                    continue
                rows.append(
                    {
                        "model": model,
                        "objective": objective,
                        "masking": wrapper_label,
                        "method": method,
                        "forget_rouge_l": metric_value(result, "forget_rouge_l"),
                        "utility": metric_value(result, "utility"),
                        "mia_auc": metric_value(result, "mia_auc"),
                    }
                )
    return rows


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def best_marks(rows: list[dict], metric: str) -> set[tuple[str, str, str]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get(metric) is None:
            continue
        grouped.setdefault((row["model"], row["objective"]), []).append(row)

    marked: set[tuple[str, str, str]] = set()
    for key, group in grouped.items():
        if metric == "forget_rouge_l":
            best = min(row[metric] for row in group)
        elif metric == "utility":
            best = max(row[metric] for row in group)
        elif metric == "mia_auc":
            best = min(abs(row[metric] - 0.5) for row in group)
            for row in group:
                if abs(row[metric] - 0.5) == best:
                    marked.add((row["model"], row["objective"], row["method"]))
            continue
        else:
            raise ValueError(metric)
        for row in group:
            if row[metric] == best:
                marked.add((row["model"], row["objective"], row["method"]))
    return marked


def make_markdown(rows: list[dict]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SWEEP_ROOT / "summary.md"
    lines = [
        "# LLM Model x Unlearning Method Default Sweep",
        "",
        "Default setting: `alpha_forget=1.5`, `beta_retain=1.0`, `probe.activity_quantile=0.10`, `probe.margin_quantile=0.05`, `mask.method=probe`, `probe.grouping=relative`.",
        "",
        "`No mask` is the pure unlearning objective. `Single mask` is `forget_only_<objective>`. `Dual mask` is `cadmu_<objective>`.",
        "",
        "Best values are marked per model and objective: Forget ROUGE-L lower is better, Utility higher is better, MIA AUC closer to 0.5 is better.",
        "",
        "| Model | Objective | Masking | Method | Forget ROUGE-L ↓ | Utility ↑ | MIA AUC ↔0.5 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    best_forget = best_marks(rows, "forget_rouge_l")
    best_utility = best_marks(rows, "utility")
    best_mia = best_marks(rows, "mia_auc")
    for row in sorted(rows, key=lambda item: (item["model"], item["objective"], item["method"])):
        key = (row["model"], row["objective"], row["method"])
        forget = fmt(row["forget_rouge_l"])
        utility = fmt(row["utility"])
        mia = fmt(row["mia_auc"])
        if key in best_forget:
            forget = f"**{forget}**"
        if key in best_utility:
            utility = f"**{utility}**"
        if key in best_mia:
            mia = f"**{mia}**"
        lines.append(
            f"| {row['model']} | {row['objective']} | {row['masking']} | `{row['method']}` | {forget} | {utility} | {mia} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (SWEEP_ROOT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {out_path}")
    print(f"saved {SWEEP_ROOT / 'summary.json'}")


def plot_objective(rows: list[dict], objective: str, metric: str, title: str) -> None:
    objective_rows = [row for row in rows if row["objective"] == objective and row.get(metric) is not None]
    models = sorted({row["model"] for row in objective_rows})
    if not models:
        return

    by_key = {(row["model"], row["masking"]): row[metric] for row in objective_rows}
    fig_width = max(10.0, len(models) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    x = list(range(len(models)))
    width = 0.25
    offsets = [-width, 0.0, width]
    for (wrapper, wrapper_label), offset in zip(WRAPPERS, offsets):
        ys = [by_key.get((model, wrapper_label), float("nan")) for model in models]
        ax.bar(
            [idx + offset for idx in x],
            ys,
            width=width,
            color=COLORS[wrapper],
            label=wrapper_label,
        )

    ax.set_title(f"{objective}: {title}")
    ax.set_ylabel("raw value")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    out_path = PLOT_DIR / f"{objective}_{metric}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    rows = collect_rows()
    if not rows:
        raise RuntimeError(f"No completed runs found under {SWEEP_ROOT}")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    make_markdown(rows)
    for objective in OBJECTIVES:
        for metric, title, direction in METRICS:
            plot_objective(rows, objective, metric, f"{title} ({direction} better)")


if __name__ == "__main__":
    main()
