from pathlib import Path

from plot_phi4_mini_instruct_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    plot(
        ROOT / "outputs/sweeps/phi4_mini_instruct/beta/mask_compare_probe_relative_beta_sweep",
        "beta",
        "beta_",
        "beta_retain",
        "Phi-4-mini probe-relative beta sweep",
    )
