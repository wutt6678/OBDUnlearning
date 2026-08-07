from pathlib import Path

from plot_smollm3_3b_probe_relative_sweeps import ROOT, plot

SWEEP_ROOT = ROOT / "outputs/sweeps/smollm3_3b/main/mask_compare_probe_relative_main_sweep"

if __name__ == "__main__":
    plot(
        SWEEP_ROOT,
        "activity",
        "activity_q_",
        "activity_quantile",
        "SmolLM3-3B probe-relative main_sweep sweep",
    )
    plot(
        SWEEP_ROOT,
        "margin",
        "margin_q_",
        "margin_quantile",
        "SmolLM3-3B probe-relative main_sweep sweep",
    )
