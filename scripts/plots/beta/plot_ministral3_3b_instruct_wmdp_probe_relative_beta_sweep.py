from pathlib import Path

from plot_benchmark_probe_relative_sweeps import ROOT, plot

if __name__ == "__main__":
    root = ROOT / "outputs/sweeps/ministral3_3b_instruct/beta/wmdp_probe_relative_beta_sweep"
    title = "ministral3 3b instruct wmdp probe-relative beta sweep"
    benchmark = "wmdp"
    for group in ["beta"]:
        if group == "alpha":
            plot(root, "alpha", "alpha_", "alpha_forget", title, benchmark=benchmark)
        elif group == "beta":
            plot(root, "beta", "beta_", "beta_retain", title, benchmark=benchmark)
        elif group == "activity":
            plot(root, "activity", "activity_q_", "activity_quantile", title, benchmark=benchmark)
        elif group == "margin":
            plot(root, "margin", "margin_q_", "margin_quantile", title, benchmark=benchmark)
