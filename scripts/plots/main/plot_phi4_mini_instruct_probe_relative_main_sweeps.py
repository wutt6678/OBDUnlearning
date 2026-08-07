from pathlib import Path

from plot_phi4_mini_instruct_probe_relative_sweeps import ROOT, plot

SWEEP_ROOT = ROOT / "outputs/sweeps/phi4_mini_instruct/main/mask_compare_probe_relative_main_sweep"

if __name__ == "__main__":
    plot(SWEEP_ROOT, "activity", "activity_q_", "activity_quantile", "Phi-4-mini probe-relative main sweep")
    plot(SWEEP_ROOT, "margin", "margin_q_", "margin_quantile", "Phi-4-mini probe-relative main sweep")
