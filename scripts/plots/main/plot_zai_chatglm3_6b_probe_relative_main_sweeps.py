from pathlib import Path

from plot_zai_chatglm3_6b_probe_relative_sweeps import ROOT, plot

SWEEP_ROOT = ROOT / "outputs/sweeps/zai_chatglm3_6b/main/mask_compare_probe_relative_main_sweep"

if __name__ == "__main__":
    plot(
        SWEEP_ROOT,
        "activity",
        "activity_q_",
        "activity_quantile",
        "ChatGLM3-6B probe-relative main_sweep sweep",
    )
    plot(
        SWEEP_ROOT,
        "margin",
        "margin_q_",
        "margin_quantile",
        "ChatGLM3-6B probe-relative main_sweep sweep",
    )
