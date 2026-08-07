from pathlib import Path

from plot_smollm3_3b_probe_relative_sweeps import ROOT, plot

if __name__ == "__main__":
    plot(
        ROOT / "outputs/sweeps/smollm3_3b/beta/mask_compare_probe_relative_beta_sweep",
        "beta",
        "beta_",
        "beta_retain",
        "SmolLM3-3B probe-relative beta_retain sweep",
    )
