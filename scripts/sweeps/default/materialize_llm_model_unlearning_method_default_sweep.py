from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "configs/llm_model_unlearning_method_default_sweep"
OUTPUT_ROOT = "outputs/sweeps/{model}/default/llm_model_unlearning_method_default_sweep"
OBJECTIVES = ["grad_diff", "grad_ascent", "npo", "dpo", "simnpo", "rmu", "kl", "retrain"]
WRAPPERS = ["none", "forget_only", "cadmu"]


def clone_cfg(obj: dict) -> dict:
    return yaml.safe_load(yaml.safe_dump(obj, sort_keys=False))


def default_template_paths() -> list[Path]:
    return sorted(ROOT.glob("configs/*_mask_compare_probe_relative_alpha_sweep/alpha/alpha_150.yaml"))


def model_slug_from_template(path: Path) -> str:
    return path.parent.parent.name.removesuffix("_mask_compare_probe_relative_alpha_sweep")


def materialize_one(template_path: Path, objective: str) -> Path:
    model_slug = model_slug_from_template(template_path)
    cfg = yaml.safe_load(template_path.read_text(encoding="utf-8"))

    cfg.setdefault("mask", {})["method"] = "probe"
    cfg.setdefault("probe", {})["grouping"] = "relative"
    cfg["probe"]["activity_quantile"] = 0.10
    cfg["probe"]["margin_quantile"] = 0.05

    unlearn = cfg.setdefault("unlearn", {})
    unlearn["objective"] = objective
    unlearn["wrappers"] = WRAPPERS
    unlearn["alpha_forget"] = 1.5
    unlearn["beta_retain"] = 1.0
    unlearn.setdefault("npo_beta", 1.0)
    unlearn.setdefault("dpo_beta", 1.0)
    unlearn.setdefault("simnpo_beta", 1.0)
    unlearn.setdefault("rmu_target_scale", 1.0)
    # Keep pure no-mask objective model-local. A global shared root would mix models.
    unlearn["shared_methods"] = []

    cfg["output_dir"] = OUTPUT_ROOT.format(model=model_slug) + f"/{objective}"
    cfg["save_models"] = False

    out_path = OUT_ROOT / f"{model_slug}_sweep" / f"{objective}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def main() -> None:
    written = []
    for template_path in default_template_paths():
        for objective in OBJECTIVES:
            written.append(materialize_one(template_path, objective))
    print(f"materialized {len(written)} configs under {OUT_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
