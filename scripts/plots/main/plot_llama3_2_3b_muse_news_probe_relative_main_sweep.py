from pathlib import Path

from plot_benchmark_probe_relative_sweeps import ROOT, plot

if __name__ == "__main__":
    root = ROOT / "outputs/sweeps/llama3_2_3b/main/muse_news_probe_relative_main_sweep"
    title = "llama3 2 3b muse news probe-relative main sweep"
    benchmark = "muse"
    for group in ["activity", "margin"]:
        if group == "alpha":
            plot(root, "alpha", "alpha_", "alpha_forget", title, benchmark=benchmark)
        elif group == "beta":
            plot(root, "beta", "beta_", "beta_retain", title, benchmark=benchmark)
        elif group == "activity":
            plot(root, "activity", "activity_q_", "activity_quantile", title, benchmark=benchmark)
        elif group == "margin":
            plot(root, "margin", "margin_q_", "margin_quantile", title, benchmark=benchmark)
