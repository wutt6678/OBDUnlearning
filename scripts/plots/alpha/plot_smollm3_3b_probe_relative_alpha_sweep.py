from pathlib import Path

from plot_smollm3_3b_probe_relative_sweeps import ROOT, plot

if __name__ == "__main__":
    plot(
        ROOT / "outputs/sweeps/smollm3_3b/alpha/mask_compare_probe_relative_alpha_sweep",
        "alpha",
        "alpha_",
        "alpha_forget",
        "SmolLM3-3B probe-relative alpha_forget sweep",
    )
