from pathlib import Path

from plot_gemma4_e2b_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    plot(
        ROOT / "outputs/gemma4_e2b_mask_compare_probe_relative_beta_sweep",
        "beta",
        "beta_",
        "beta_retain",
        "Gemma4-E2B probe-relative beta sweep",
    )
