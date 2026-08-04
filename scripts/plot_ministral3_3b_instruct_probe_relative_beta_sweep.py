from pathlib import Path

from plot_ministral3_3b_instruct_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    plot(
        ROOT / "outputs/ministral3_3b_instruct_mask_compare_probe_relative_beta_sweep",
        "beta",
        "beta_",
        "beta_retain",
        "Ministral3-3B probe-relative beta sweep",
    )
