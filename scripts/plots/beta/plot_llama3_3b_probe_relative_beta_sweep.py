from pathlib import Path

from plot_llama3_3b_probe_relative_sweeps import plot

ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    plot(
        ROOT / "outputs/sweeps/llama3_3b/beta/mask_compare_probe_relative_beta_sweep",
        "beta",
        "beta_",
        "beta_retain",
        [
            "grad_diff",
            "forget_only_grad_diff",
            "cadmu_grad_diff",
                ],
    )
