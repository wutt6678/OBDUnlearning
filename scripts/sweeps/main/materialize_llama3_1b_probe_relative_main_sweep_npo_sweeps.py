from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BASE_CFG = ROOT / "configs/llama3_1b_mask_compare_probe_relative_main_sweep.yaml"
OUT_ROOT = (
    ROOT
    / "configs/llama3_1b_mask_compare_probe_relative_main_sweep_npo_sweep"
)
OUTPUT_ROOT = (
    "outputs/sweeps/llama3_1b/main/mask_compare_probe_relative_main_sweep_npo_sweep"
)

VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

def dump_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

def main() -> None:
    with BASE_CFG.open("r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    base["unlearn"]["objective"] = "npo"
    base["unlearn"]["method_overrides"] = {
    }
    base["unlearn"]["shared_methods"] = ["npo"]

    for value in VALUES:
        cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
        cfg["probe"]["activity_quantile"] = value
        cfg["probe"]["margin_quantile"] = 0.05
        stem = f"activity_q_{int(round(value * 100)):02d}"
        cfg["output_dir"] = f"{OUTPUT_ROOT}/activity/{stem}"
        dump_config(OUT_ROOT / "activity" / f"{stem}.yaml", cfg)

    for value in VALUES:
        cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
        cfg["probe"]["activity_quantile"] = 0.1
        cfg["probe"]["margin_quantile"] = value
        stem = f"margin_q_{int(round(value * 100)):02d}"
        cfg["output_dir"] = f"{OUTPUT_ROOT}/margin/{stem}"
        dump_config(OUT_ROOT / "margin" / f"{stem}.yaml", cfg)

if __name__ == "__main__":
    main()
