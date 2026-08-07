from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs/mistral7b_instruct_v03_mask_compare_probe_relative_main_sweep.yaml"
VALUES = [step / 2 for step in range(1, 11)]
QUANTILES = [step / 20 for step in range(11)]

def clone(obj: dict) -> dict:
    return yaml.safe_load(yaml.safe_dump(obj, sort_keys=False))

def write(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

def main() -> None:
    with BASE_CONFIG.open("r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)

    for value in VALUES:
        stem = f"alpha_{int(round(value * 100)):03d}"
        cfg = clone(base)
        cfg["unlearn"]["alpha_forget"] = value
        cfg["unlearn"]["beta_retain"] = 1.0
        cfg["unlearn"]["shared_methods"] = []
        cfg["output_dir"] = (
            f"outputs/sweeps/mistral7b_instruct_v03/alpha/mask_compare_probe_relative_alpha_sweep/alpha/{stem}"
        )
        write(
            ROOT / "configs/mistral7b_instruct_v03_mask_compare_probe_relative_alpha_sweep/alpha"
            / f"{stem}.yaml",
            cfg,
        )

    for value in VALUES:
        stem = f"beta_{int(round(value * 100)):03d}"
        cfg = clone(base)
        cfg["unlearn"]["alpha_forget"] = 1.5
        cfg["unlearn"]["beta_retain"] = value
        cfg["unlearn"].pop("method_overrides", None)
        cfg["unlearn"]["shared_methods"] = []
        cfg["output_dir"] = (
            f"outputs/sweeps/mistral7b_instruct_v03/beta/mask_compare_probe_relative_beta_sweep/beta/{stem}"
        )
        write(
            ROOT / "configs/mistral7b_instruct_v03_mask_compare_probe_relative_beta_sweep/beta"
            / f"{stem}.yaml",
            cfg,
        )

    sweep_config_root = ROOT / "configs/mistral7b_instruct_v03_mask_compare_probe_relative_main_sweep"
    sweep_output_root = "outputs/sweeps/mistral7b_instruct_v03/main/mask_compare_probe_relative_main_sweep"
    for group, values in (("activity", QUANTILES), ("margin", QUANTILES)):
        for value in values:
            cfg = clone(base)
            cfg["unlearn"]["wrappers"] = [
                "none",
                "forget_only",
                "cadmu",
            ]
            cfg["unlearn"]["shared_methods"] = ["grad_diff"]
            if group == "activity":
                cfg["probe"]["activity_quantile"] = value
                cfg["probe"]["margin_quantile"] = 0.05
            else:
                cfg["probe"]["activity_quantile"] = 0.1
                cfg["probe"]["margin_quantile"] = value
            stem = f"{group}_q_{int(round(value * 100)):02d}"
            cfg["output_dir"] = f"{sweep_output_root}/{group}/{stem}"
            write(sweep_config_root / group / f"{stem}.yaml", cfg)

if __name__ == "__main__":
    main()
