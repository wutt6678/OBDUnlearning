from pathlib import Path

from plot_qwen3_5_9b_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    plot(
        ROOT / "outputs/sweeps/qwen3_5_9b/beta/mask_compare_probe_relative_beta_sweep",
        "beta",
        "beta_",
        "beta_retain",
        "Qwen3.5-8B probe-relative beta sweep",
    )
