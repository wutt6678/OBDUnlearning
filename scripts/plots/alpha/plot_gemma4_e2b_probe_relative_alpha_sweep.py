from pathlib import Path

from plot_gemma4_e2b_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    plot(
        ROOT / "outputs/sweeps/gemma4_e2b/alpha/mask_compare_probe_relative_alpha_sweep",
        "alpha",
        "alpha_",
        "alpha_forget",
        "Gemma4-E2B probe-relative alpha sweep",
    )
