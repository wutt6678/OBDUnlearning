from pathlib import Path

from plot_zai_glm_4_9b_probe_relative_sweeps import ROOT, plot

if __name__ == "__main__":
    plot(
        ROOT / "outputs/zai_glm_4_9b_mask_compare_probe_relative_alpha_sweep",
        "alpha",
        "alpha_",
        "alpha_forget",
        "GLM-4-9B probe-relative alpha_forget sweep",
    )
