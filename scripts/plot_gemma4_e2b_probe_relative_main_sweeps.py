from pathlib import Path

from plot_gemma4_e2b_probe_relative_sweeps import ROOT, plot

SWEEP_ROOT = ROOT / "outputs/gemma4_e2b_mask_compare_probe_relative_main_sweep"

if __name__ == "__main__":
    plot(SWEEP_ROOT, "activity", "activity_q_", "activity_quantile", "Gemma4-E2B probe-relative main sweep")
    plot(SWEEP_ROOT, "margin", "margin_q_", "margin_quantile", "Gemma4-E2B probe-relative main sweep")
