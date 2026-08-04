from pathlib import Path

from plot_qwen3_5_4b_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    plot(
        ROOT / "outputs/qwen3_5_4b_mask_compare_probe_relative_beta_sweep",
        "beta",
        "beta_",
        "beta_retain",
        "Qwen3.5-4B probe-relative beta sweep",
    )
