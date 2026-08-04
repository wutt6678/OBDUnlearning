from pathlib import Path

from plot_zai_glm_4_9b_probe_relative_sweeps import ROOT, plot

SWEEP_ROOT = ROOT / "outputs/zai_glm_4_9b_mask_compare_probe_relative_main_sweep"

if __name__ == "__main__":
    plot(
        SWEEP_ROOT,
        "activity",
        "activity_q_",
        "activity_quantile",
        "GLM-4-9B probe-relative main_sweep sweep",
    )
    plot(
        SWEEP_ROOT,
        "margin",
        "margin_q_",
        "margin_quantile",
        "GLM-4-9B probe-relative main_sweep sweep",
    )
