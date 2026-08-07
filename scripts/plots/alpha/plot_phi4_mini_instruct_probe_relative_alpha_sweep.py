from pathlib import Path

from plot_phi4_mini_instruct_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    plot(
        ROOT / "outputs/sweeps/phi4_mini_instruct/alpha/mask_compare_probe_relative_alpha_sweep",
        "alpha",
        "alpha_",
        "alpha_forget",
        "Phi-4-mini probe-relative alpha sweep",
    )
