from pathlib import Path

from plot_zai_chatglm3_6b_probe_relative_sweeps import ROOT, plot

if __name__ == "__main__":
    plot(
        ROOT / "outputs/zai_chatglm3_6b_mask_compare_probe_relative_alpha_sweep",
        "alpha",
        "alpha_",
        "alpha_forget",
        "ChatGLM3-6B probe-relative alpha_forget sweep",
    )
