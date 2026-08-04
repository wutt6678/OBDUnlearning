from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = ROOT / "configs/llama3_1b_mask_compare_probe_relative_main_sweep.yaml"
OUT_ROOT = ROOT / "configs/llama3_1b_mask_compare_probe_relative_main_sweep"

ACTIVITY_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MARGIN_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

def load_base_config() -> dict:
    with BASE_CFG.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def dump_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

def make_output_dir(group: str, stem: str) -> str:
    return f"outputs/llama3_1b_mask_compare_probe_relative_main_sweep/{group}/{stem}"

def main() -> None:
    base = load_base_config()

    for value in ACTIVITY_VALUES:
        cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
        cfg["probe"]["activity_quantile"] = value
        cfg["probe"]["margin_quantile"] = base["probe"]["margin_quantile"]
        stem = f"activity_q_{int(round(value * 100)):02d}"
        cfg["output_dir"] = make_output_dir("activity", stem)
        dump_config(OUT_ROOT / "activity" / f"{stem}.yaml", cfg)

    for value in MARGIN_VALUES:
        cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
        cfg["probe"]["activity_quantile"] = base["probe"]["activity_quantile"]
        cfg["probe"]["margin_quantile"] = value
        stem = f"margin_q_{int(round(value * 100)):02d}"
        cfg["output_dir"] = make_output_dir("margin", stem)
        dump_config(OUT_ROOT / "margin" / f"{stem}.yaml", cfg)

if __name__ == "__main__":
    main()
