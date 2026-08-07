from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = ROOT / "configs/llama3_1b_mask_compare_probe_relative_main_sweep.yaml"
CONFIG_ROOT = ROOT / "configs/llama3_1b_mask_construction_usage_compare"
OUTPUT_ROOT = "outputs/sweeps/llama3_1b/masks/mask_construction_usage_compare"

CONSTRUCTIONS = [
    ("saliency_grad_abs", {"mask_method": "saliency", "saliency_mode": "grad_abs"}),
    ("saliency_weight_grad", {"mask_method": "saliency", "saliency_mode": "weight_grad"}),
    ("saliency_fisher", {"mask_method": "saliency", "saliency_mode": "fisher"}),
    ("probe_absolute", {"mask_method": "probe", "probe_grouping": "absolute"}),
    ("probe_relative", {"mask_method": "probe", "probe_grouping": "relative"}),
]

def clone(obj: dict) -> dict:
    return yaml.safe_load(yaml.safe_dump(obj, sort_keys=False))

def main() -> None:
    with BASE_CONFIG.open("r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)

    for name, opts in CONSTRUCTIONS:
        cfg = clone(base)
        cfg["mask"]["method"] = opts["mask_method"]
        if opts["mask_method"] == "saliency":
            cfg["saliency"]["mode"] = opts["saliency_mode"]
        else:
            cfg["probe"]["grouping"] = opts["probe_grouping"]
            if opts["probe_grouping"] == "relative":
                cfg["probe"]["activity_quantile"] = 0.1
                cfg["probe"]["margin_quantile"] = 0.05
        cfg["unlearn"]["wrappers"] = [
            "none",
            "forget_only",
            "cadmu",
                    ]
        cfg["unlearn"]["shared_methods"] = ["grad_diff"]
        cfg["output_dir"] = f"{OUTPUT_ROOT}/{name}"
        out = CONFIG_ROOT / f"{name}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

if __name__ == "__main__":
    main()
