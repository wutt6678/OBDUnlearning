from pathlib import Path

from plot_qwen3_5_4b_probe_relative_sweeps import ROOT, plot

SWEEP_ROOT = ROOT / "outputs/qwen3_5_4b_mask_compare_probe_relative_main_sweep"

if __name__ == "__main__":
    plot(SWEEP_ROOT, "activity", "activity_q_", "activity_quantile", "Qwen3.5-4B probe-relative main sweep")
    plot(SWEEP_ROOT, "margin", "margin_q_", "margin_quantile", "Qwen3.5-4B probe-relative main sweep")
