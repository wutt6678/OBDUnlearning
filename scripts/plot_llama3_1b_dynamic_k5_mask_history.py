from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "outputs/llama3_1b_mask_compare_probe_relative_dynamic_k5_alpha_sweep"
PLOT_DIR = SWEEP_ROOT / "plots" / "mask_history"
GROUPS = ["retain_only", "forget_only", "conflict", "neutral", "retain", "forget"]


def load_history(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_history(history_path: Path) -> None:
    history = load_history(history_path)
    if not history:
        return
    run_dir = history_path.parents[1]
    method = history_path.parent.name
    alpha_name = run_dir.name
    steps = [item["step"] for item in history]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for group in GROUPS:
        ys = [item.get("stats", {}).get(group) for item in history]
        if all(value is None for value in ys):
            continue
        ax.plot(steps, ys, marker="o", linewidth=2.0, markersize=4, label=group)

    ax.set_title(f"Dynamic mask ratios: {alpha_name} / {method}")
    ax.set_xlabel("unlearning step")
    ax.set_ylabel("mask ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    out_dir = PLOT_DIR / alpha_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{method}_mask_history.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def write_summary(paths: list[Path]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dynamic Mask History Summary",
        "",
        "| alpha | method | step | retain_only | forget_only | conflict | neutral | retain | forget |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for path in paths:
        history = load_history(path)
        alpha = path.parents[1].name
        method = path.parent.name
        for item in history:
            stats = item.get("stats", {})
            lines.append(
                f"| {alpha} | `{method}` | {item['step']} | "
                f"{stats.get('retain_only', float('nan')):.4f} | "
                f"{stats.get('forget_only', float('nan')):.4f} | "
                f"{stats.get('conflict', float('nan')):.4f} | "
                f"{stats.get('neutral', float('nan')):.4f} | "
                f"{stats.get('retain', float('nan')):.4f} | "
                f"{stats.get('forget', float('nan')):.4f} |"
            )
    out_path = PLOT_DIR / "dynamic_mask_history_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out_path}")


def main() -> None:
    paths = sorted(SWEEP_ROOT.glob("alpha/alpha_*/*/dynamic_mask_history.json"))
    if not paths:
        raise RuntimeError(f"No dynamic mask histories found under {SWEEP_ROOT}")
    for path in paths:
        plot_history(path)
    write_summary(paths)


if __name__ == "__main__":
    main()
