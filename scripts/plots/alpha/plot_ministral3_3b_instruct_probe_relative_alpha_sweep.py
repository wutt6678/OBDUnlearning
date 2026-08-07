from pathlib import Path

from plot_ministral3_3b_instruct_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    plot(
        ROOT / "outputs/sweeps/ministral3_3b_instruct/alpha/mask_compare_probe_relative_alpha_sweep",
        "alpha",
        "alpha_",
        "alpha_forget",
        "Ministral3-3B probe-relative alpha sweep",
    )
