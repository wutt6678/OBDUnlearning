from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "outputs/zai_chatglm3_6b_mask_compare_probe_relative_main_sweep"
PLOT_DIR = SWEEP_ROOT / "plots"
GROUPS = [
    ("activity", "activity_q_*", "activity_quantile"),
    ("margin", "margin_q_*", "margin_quantile"),
]
ORDER = ["retain_only", "forget_only", "conflict", "neutral"]
COLORS = {
    "retain_only": "#1f77b4",
    "forget_only": "#ff7f0e",
    "conflict": "#9467bd",
    "neutral": "#8c564b",
}

def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for group, pattern, x_label in GROUPS:
        paths = sorted((SWEEP_ROOT / group).glob(f"{pattern}/mask_stats.json"))
        if not paths:
            print(f"skipping {group}: no mask_stats found")
            continue
        rows = []
        for path in paths:
            x = int(path.parent.name.split("_")[-1]) / 100.0
            rows.append((x, json.loads(path.read_text(encoding="utf-8"))))
        rows.sort(key=lambda item: item[0])

        xs = [x for x, _ in rows]
        bottoms = [0.0] * len(rows)
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        for key in ORDER:
            values = [stats[key] for _, stats in rows]
            tops = [bottom + value for bottom, value in zip(bottoms, values)]
            ax.fill_between(xs, bottoms, tops, color=COLORS[key], alpha=0.7, label=key)
            ax.plot(xs, tops, color=COLORS[key], linewidth=1.5)
            bottoms = tops

        ax.set_title(f"ChatGLM3-6B probe-relative main_sweep: {group} neuron split ratios")
        ax.set_xlabel(x_label)
        ax.set_ylabel("ratio")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        output = PLOT_DIR / f"{group}_neuron_split_ratios.png"
        fig.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {output}")

if __name__ == "__main__":
    main()
