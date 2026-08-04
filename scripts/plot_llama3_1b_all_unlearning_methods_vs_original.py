from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = ROOT / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep_all_methods" / "plots"
DEFAULT_GROUP = "activity"
DEFAULT_RUN = "activity_q_10"

OBJECTIVE_ROOTS = {
    "grad_diff": ROOT / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep",
    "grad_ascent": ROOT / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep_grad_ascent_sweep",
    "npo": ROOT / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep_npo_sweep",
    "dpo": ROOT / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep_dpo_sweep",
    "simnpo": ROOT / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep_simnpo_sweep",
    "rmu": ROOT / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep_rmu_sweep",
    "kl": ROOT / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep_kl_sweep",
    "retrain": ROOT / "outputs/llama3_1b_mask_compare_probe_relative_main_sweep_retrain_sweep",
}
WRAPPERS = [
    ("none", "no mask"),
    ("forget_only", "single mask"),
    ("cadmu", "dual mask"),
]
METRICS = [
    ("forget_rouge_l", "Forget ROUGE-L (lower better)"),
    ("utility_score", "Utility score (higher better)"),
    ("privacy", "MIA AUC (lower better)"),
]
COLORS = {
    "no mask": "#1f77b4",
    "single mask": "#2ca02c",
    "dual mask": "#9467bd",
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


def load_run(root: Path) -> dict | None:
    path = root / DEFAULT_GROUP / DEFAULT_RUN / "open_tofu_comparison.json"
    if not path.exists():
        print(f"missing: {path}")
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_rows() -> tuple[dict | None, list[dict]]:
    original = None
    rows = []
    for objective, root in OBJECTIVE_ROOTS.items():
        run = load_run(root)
        if run is None:
            continue
        if original is None:
            original = run["original"]
        for wrapper, wrapper_label in WRAPPERS:
            method = method_name(objective, wrapper)
            if method not in run.get("methods", {}):
                print(f"missing method {method} in {root.name}")
                continue
            rows.append(
                {
                    "objective": objective,
                    "wrapper": wrapper,
                    "wrapper_label": wrapper_label,
                    "method": method,
                    "result": run["methods"][method],
                }
            )
    return original, rows


def plot_all_vs_original() -> None:
    original, rows = collect_rows()
    if original is None or not rows:
        raise RuntimeError("No completed runs found for all-method comparison")

    objectives = list(dict.fromkeys(row["objective"] for row in rows))
    x = np.arange(len(objectives))
    width = 0.24
    offsets = {"none": -width, "forget_only": 0.0, "cadmu": width}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 5.2), sharex=True)
    for ax, (metric_key, title) in zip(axes, METRICS):
        for wrapper, wrapper_label in WRAPPERS:
            values = []
            positions = []
            for i, objective in enumerate(objectives):
                match = next((row for row in rows if row["objective"] == objective and row["wrapper"] == wrapper), None)
                if match is None:
                    continue
                positions.append(i + offsets[wrapper])
                values.append(metric(match["result"], metric_key))
            ax.bar(positions, values, width, color=COLORS[wrapper_label], label=wrapper_label)
        ax.axhline(metric(original, metric_key), color="#d62728", linestyle="--", linewidth=2, label="original")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(objectives, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("raw value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Llama3-1B all unlearning methods vs original ({DEFAULT_GROUP}/{DEFAULT_RUN})", y=1.02)
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "all_methods_vs_original_forget_rouge_l_utility_score_mia_auc.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    plot_all_vs_original()


if __name__ == "__main__":
    main()
