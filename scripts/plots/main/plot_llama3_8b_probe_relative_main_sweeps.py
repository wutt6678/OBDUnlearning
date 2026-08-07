from pathlib import Path

from plot_llama3_8b_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[3]
SWEEP_ROOT = ROOT / "outputs/sweeps/llama3_8b/main/mask_compare_probe_relative_main_sweep"

if __name__ == "__main__":
    methods = ["grad_diff", "forget_only_grad_diff", "cadmu_grad_diff"]
    plot(SWEEP_ROOT, "activity", "activity_q_", "activity_quantile", methods)
    plot(SWEEP_ROOT, "margin", "margin_q_", "margin_quantile", methods)
