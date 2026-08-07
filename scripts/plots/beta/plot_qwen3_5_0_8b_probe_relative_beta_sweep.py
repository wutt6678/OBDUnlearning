from pathlib import Path

from plot_qwen3_5_0_8b_probe_relative_sweeps import ROOT, plot

if __name__ == "__main__":
    plot(
        ROOT / "outputs/sweeps/qwen3_5_0_8b/beta/mask_compare_probe_relative_beta_sweep",
        "beta",
        "beta_",
        "beta_retain",
        "Qwen3.5-0.8B probe-relative beta_retain sweep",
    )
