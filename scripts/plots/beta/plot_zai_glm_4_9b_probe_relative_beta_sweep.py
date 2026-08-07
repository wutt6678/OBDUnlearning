from pathlib import Path

from plot_zai_glm_4_9b_probe_relative_sweeps import ROOT, plot

if __name__ == "__main__":
    plot(
        ROOT / "outputs/sweeps/zai_glm_4_9b/beta/mask_compare_probe_relative_beta_sweep",
        "beta",
        "beta_",
        "beta_retain",
        "GLM-4-9B probe-relative beta_retain sweep",
    )
