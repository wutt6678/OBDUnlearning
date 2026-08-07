from pathlib import Path

from plot_zai_chatglm3_6b_probe_relative_sweeps import ROOT, plot

if __name__ == "__main__":
    plot(
        ROOT / "outputs/sweeps/zai_chatglm3_6b/beta/mask_compare_probe_relative_beta_sweep",
        "beta",
        "beta_",
        "beta_retain",
        "ChatGLM3-6B probe-relative beta_retain sweep",
    )
