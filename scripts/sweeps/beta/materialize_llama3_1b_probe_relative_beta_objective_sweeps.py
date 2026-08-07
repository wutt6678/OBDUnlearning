from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = ROOT / "configs/llama3_1b_mask_compare_probe_relative_main_sweep.yaml"
CONFIG_ROOT = ROOT / "configs/llama3_1b_mask_compare_probe_relative_beta_objective_sweeps"
OUTPUT_ROOT = "outputs/sweeps/llama3_1b/beta/mask_compare_probe_relative_beta_objective_sweeps"

OBJECTIVES = [
    "grad_diff",
    "grad_ascent",
    "npo",
    "dpo",
    "simnpo",
    "rmu",
    "kl",
    "retrain",
]
BETA_VALUES = [step / 2 for step in range(1, 21)]


def clone(obj: dict) -> dict:
    return yaml.safe_load(yaml.safe_dump(obj, sort_keys=False))


def main() -> None:
    with BASE_CONFIG.open("r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)

    for objective in OBJECTIVES:
        for value in BETA_VALUES:
            cfg = clone(base)
            cfg["probe"]["activity_quantile"] = 0.1
            cfg["probe"]["margin_quantile"] = 0.05
            cfg["unlearn"]["objective"] = objective
            cfg["unlearn"]["wrappers"] = ["none", "forget_only", "cadmu"]
            cfg["unlearn"]["steps"] = 50
            cfg["unlearn"]["lr"] = 0.0001
            cfg["unlearn"]["alpha_forget"] = 1.5
            cfg["unlearn"]["beta_retain"] = value
            cfg["unlearn"].setdefault("npo_beta", 1.0)
            cfg["unlearn"].setdefault("projection_eps", 1.0e-12)
            cfg["unlearn"].setdefault("dpo_beta", 1.0)
            cfg["unlearn"].setdefault("simnpo_beta", 1.0)
            cfg["unlearn"].setdefault("rmu_target_scale", 1.0)
            # beta changes every objective/wrapper; do not share no-mask across beta values.
            cfg["unlearn"]["shared_methods"] = []

            stem = f"beta_{int(round(value * 100)):03d}"
            sweep_name = f"{objective}_beta_sweep"
            cfg["output_dir"] = f"{OUTPUT_ROOT}/{sweep_name}/beta/{stem}"
            path = CONFIG_ROOT / sweep_name / "beta" / f"{stem}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False)


if __name__ == "__main__":
    main()
