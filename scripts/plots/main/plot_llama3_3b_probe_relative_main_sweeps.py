from pathlib import Path

from plot_llama3_3b_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[3]
SWEEP_ROOT = (
    ROOT
    / "outputs/sweeps/llama3_3b/main/mask_compare_probe_relative_main_sweep"
)
METHODS = [
    "grad_diff",
    "forget_only_grad_diff",
    "cadmu_grad_diff",
]

if __name__ == "__main__":
    plot(
        SWEEP_ROOT,
        "activity",
        "activity_q_",
        "activity_quantile",
        METHODS,
    )
    plot(
        SWEEP_ROOT,
        "margin",
        "margin_q_",
        "margin_quantile",
        METHODS,
    )
