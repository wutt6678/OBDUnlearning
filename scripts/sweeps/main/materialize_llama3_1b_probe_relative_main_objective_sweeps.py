from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BASE_CFG = ROOT / "configs/llama3_1b_mask_compare_probe_relative_main_sweep.yaml"
OUT_ROOT = ROOT / "configs/llama3_1b_mask_compare_probe_relative_main_objective_sweeps"
OUTPUT_ROOT = "outputs/sweeps/llama3_1b/main/mask_compare_probe_relative_main_objective_sweeps"
OBJECTIVES = ["grad_diff", "grad_ascent", "npo", "dpo", "simnpo", "rmu", "kl", "retrain"]
VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def dump_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def make_base(objective: str) -> dict:
    base = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    base["unlearn"]["objective"] = objective
    base["unlearn"]["wrappers"] = ["none", "forget_only", "cadmu"]
    base["unlearn"]["shared_methods"] = [objective]
    base["unlearn"].setdefault("dpo_beta", 1.0)
    base["unlearn"].setdefault("simnpo_beta", 1.0)
    base["unlearn"].setdefault("rmu_target_scale", 1.0)
    return base


def materialize_objective(objective: str) -> None:
    base = make_base(objective)
    obj_root = OUT_ROOT / f"{objective}_main_sweep"
    output_root = f"{OUTPUT_ROOT}/{objective}_main_sweep"

    for value in VALUES:
        cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
        cfg["probe"]["activity_quantile"] = value
        cfg["probe"]["margin_quantile"] = 0.05
        stem = f"activity_q_{int(round(value * 100)):02d}"
        cfg["output_dir"] = f"{output_root}/activity/{stem}"
        dump_config(obj_root / "activity" / f"{stem}.yaml", cfg)

    for value in VALUES:
        cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
        cfg["probe"]["activity_quantile"] = 0.10
        cfg["probe"]["margin_quantile"] = value
        stem = f"margin_q_{int(round(value * 100)):02d}"
        cfg["output_dir"] = f"{output_root}/margin/{stem}"
        dump_config(obj_root / "margin" / f"{stem}.yaml", cfg)


def main() -> None:
    for objective in OBJECTIVES:
        materialize_objective(objective)


if __name__ == "__main__":
    main()
