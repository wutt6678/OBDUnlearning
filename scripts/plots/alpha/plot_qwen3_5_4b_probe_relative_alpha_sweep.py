from pathlib import Path

from plot_qwen3_5_4b_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    plot(
        ROOT / "outputs/sweeps/qwen3_5_4b/alpha/mask_compare_probe_relative_alpha_sweep",
        "alpha",
        "alpha_",
        "alpha_forget",
        "Qwen3.5-4B probe-relative alpha sweep",
    )
