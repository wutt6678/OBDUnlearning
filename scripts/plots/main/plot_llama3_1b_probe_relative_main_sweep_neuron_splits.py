from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SWEEP_ROOT = ROOT / "outputs/sweeps/llama3_1b/main/mask_compare_probe_relative_main_sweep"
PLOT_DIR = SWEEP_ROOT / "plots"

GROUPS = [
    (
        "activity",
        sorted((SWEEP_ROOT / "activity").glob("activity_q_*/mask_stats.json")),
        "activity_quantile",
    ),
    (
        "margin",
        sorted((SWEEP_ROOT / "margin").glob("margin_q_*/mask_stats.json")),
        "margin_quantile",
    ),
]

COLORS = {
    "retain_only": "#1f77b4",
    "forget_only": "#ff7f0e",
    "conflict": "#9467bd",
    "neutral": "#8c564b",
}
ORDER = ["retain_only", "forget_only", "conflict", "neutral"]
LABELS = {
    "retain_only": "retain_only",
    "forget_only": "forget_only",
    "conflict": "conflict",
    "neutral": "neutral",
}

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def parse_x(path: Path) -> float:
    stem = path.parent.name
    return int(stem.split("_")[-1]) / 100.0

def collect(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        row = load_json(path)
        rows.append({"x": parse_x(path), **row})
    rows.sort(key=lambda item: item["x"])
    return rows

def plot_group(kind: str, rows: list[dict], x_label: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    xs = [row["x"] for row in rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bottoms = [0.0] * len(rows)
    for key in ORDER:
        vals = [row[key] for row in rows]
        ax.fill_between(xs, bottoms, [b + v for b, v in zip(bottoms, vals)], color=COLORS[key], alpha=0.7, label=LABELS[key])
        ax.plot(xs, [b + v for b, v in zip(bottoms, vals)], color=COLORS[key], linewidth=1.5)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_title(f"Llama3-1B probe-relative main_sweep: {kind} neuron split ratios")
    ax.set_xlabel(x_label)
    ax.set_ylabel("ratio")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out_path = PLOT_DIR / f"{kind}_neuron_split_ratios.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

def main() -> None:
    for kind, paths, x_label in GROUPS:
        rows = collect(paths)
        if not rows:
            raise RuntimeError(f"No mask_stats found for {kind}")
        plot_group(kind, rows, x_label)

if __name__ == "__main__":
    main()
