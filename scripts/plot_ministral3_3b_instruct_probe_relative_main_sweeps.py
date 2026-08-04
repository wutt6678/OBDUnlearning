from pathlib import Path

from plot_ministral3_3b_instruct_probe_relative_sweeps import ROOT, plot

SWEEP_ROOT = ROOT / "outputs/ministral3_3b_instruct_mask_compare_probe_relative_main_sweep"

if __name__ == "__main__":
    plot(SWEEP_ROOT, "activity", "activity_q_", "activity_quantile", "Ministral3-3B probe-relative main sweep")
    plot(SWEEP_ROOT, "margin", "margin_q_", "margin_quantile", "Ministral3-3B probe-relative main sweep")
