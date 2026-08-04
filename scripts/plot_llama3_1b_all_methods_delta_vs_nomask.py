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
    ("forget_only", "single mask", "#2ca02c"),
    ("cadmu", "dual mask", "#9467bd"),
]
METRICS = [
    ("forget_rouge_l", "Delta Forget ROUGE-L vs no mask"),
    ("utility_score", "Delta Utility vs no mask"),
    ("privacy", "Delta MIA AUC vs no mask"),
]


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


def collect_rows() -> list[dict]:
    rows = []
    for objective, root in OBJECTIVE_ROOTS.items():
        run = load_run(root)
        if run is None:
            continue
        no_mask_method = objective
        if no_mask_method not in run.get("methods", {}):
            print(f"missing no-mask method {no_mask_method} in {root.name}")
            continue
        no_mask = run["methods"][no_mask_method]
        for wrapper, wrapper_label, color in WRAPPERS:
            method = method_name(objective, wrapper)
            if method not in run.get("methods", {}):
                print(f"missing method {method} in {root.name}")
                continue
            rows.append(
                {
                    "objective": objective,
                    "wrapper": wrapper,
                    "wrapper_label": wrapper_label,
                    "color": color,
                    "method": method,
                    "result": run["methods"][method],
                    "no_mask": no_mask,
                }
            )
    return rows


def write_delta_summary(rows: list[dict]) -> None:
    lines = [
        "# Delta vs No-Mask Summary",
        "",
        f"Default point: `{DEFAULT_GROUP}/{DEFAULT_RUN}`.",
        "",
        "| objective | method | delta Forget ROUGE-L | delta Utility | delta MIA AUC |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['objective']} | `{row['method']}` | "
            f"{metric(row['result'], 'forget_rouge_l') - metric(row['no_mask'], 'forget_rouge_l'):+.4f} | "
            f"{metric(row['result'], 'utility_score') - metric(row['no_mask'], 'utility_score'):+.4f} | "
            f"{metric(row['result'], 'privacy') - metric(row['no_mask'], 'privacy'):+.4f} |"
        )
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "all_methods_delta_vs_nomask_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out_path}")


def plot_delta_vs_nomask() -> None:
    rows = collect_rows()
    if not rows:
        raise RuntimeError("No completed runs found for delta-vs-nomask comparison")

    objectives = list(dict.fromkeys(row["objective"] for row in rows))
    x = np.arange(len(objectives))
    width = 0.34
    offsets = {"forget_only": -width / 2, "cadmu": width / 2}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 5.2), sharex=True)
    for ax, (metric_key, title) in zip(axes, METRICS):
        for wrapper, wrapper_label, color in WRAPPERS:
            values = []
            positions = []
            for i, objective in enumerate(objectives):
                match = next((row for row in rows if row["objective"] == objective and row["wrapper"] == wrapper), None)
                if match is None:
                    continue
                positions.append(i + offsets[wrapper])
                values.append(metric(match["result"], metric_key) - metric(match["no_mask"], metric_key))
            ax.bar(positions, values, width, color=color, label=wrapper_label)
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(objectives, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("delta value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Llama3-1B all methods: mask variants delta vs no mask ({DEFAULT_GROUP}/{DEFAULT_RUN})", y=1.02)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "all_methods_delta_vs_nomask_forget_rouge_l_utility_score_mia_auc.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    write_delta_summary(rows)


def main() -> None:
    plot_delta_vs_nomask()


if __name__ == "__main__":
    main()
