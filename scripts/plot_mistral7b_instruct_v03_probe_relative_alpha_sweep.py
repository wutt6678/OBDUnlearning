from pathlib import Path

from plot_mistral7b_instruct_v03_probe_relative_sweeps import ROOT, plot

if __name__ == "__main__":
    plot(
        ROOT / "outputs/mistral7b_instruct_v03_mask_compare_probe_relative_alpha_sweep",
        "alpha",
        "alpha_",
        "alpha_forget",
        "Mistral-7B-Instruct-v0.3 probe-relative alpha_forget sweep",
    )
