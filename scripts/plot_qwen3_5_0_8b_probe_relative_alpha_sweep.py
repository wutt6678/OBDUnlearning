from pathlib import Path

from plot_qwen3_5_0_8b_probe_relative_sweeps import ROOT, plot

if __name__ == "__main__":
    plot(
        ROOT / "outputs/qwen3_5_0_8b_mask_compare_probe_relative_alpha_sweep",
        "alpha",
        "alpha_",
        "alpha_forget",
        "Qwen3.5-0.8B probe-relative alpha_forget sweep",
    )
