from pathlib import Path

from plot_phi4_mini_instruct_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    plot(
        ROOT / "outputs/phi4_mini_instruct_mask_compare_probe_relative_beta_sweep",
        "beta",
        "beta_",
        "beta_retain",
        "Phi-4-mini probe-relative beta sweep",
    )
