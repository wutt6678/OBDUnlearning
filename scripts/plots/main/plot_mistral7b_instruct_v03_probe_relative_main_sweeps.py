from pathlib import Path

from plot_mistral7b_instruct_v03_probe_relative_sweeps import ROOT, plot

SWEEP_ROOT = ROOT / "outputs/sweeps/mistral7b_instruct_v03/main/mask_compare_probe_relative_main_sweep"

if __name__ == "__main__":
    plot(
        SWEEP_ROOT,
        "activity",
        "activity_q_",
        "activity_quantile",
        "Mistral-7B-Instruct-v0.3 probe-relative main_sweep sweep",
    )
    plot(
        SWEEP_ROOT,
        "margin",
        "margin_q_",
        "margin_quantile",
        "Mistral-7B-Instruct-v0.3 probe-relative main_sweep sweep",
    )
