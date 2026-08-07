"""Materialize baseline-comparison configs and run scripts for all models.

For every model, this generates configs that run ALL base unlearning
objectives (grad_ascent, grad_diff, npo, dpo, simnpo, rmu, kl, retrain)
with all three mask wrappers (none, forget_only, cadmu) = 24 methods,
under two forget-strength settings:

  * strong: alpha_forget=1.5, beta_retain=1.0
  * weak:   alpha_forget=1.0, beta_retain=1.5

Datasets:
  * TOFU with forget01/forget05/forget10 splits
  * MUSE-Books, MUSE-News, WMDP (single native forget split each)

Configs land in configs/baselines/<model>/ and outputs are written to
outputs/baselines/<model>/<dataset>[_<split>]_<setting>. One run script
per model is written to scripts/baselines/run_<model>_baselines.sh.
"""

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs"
BASELINE_ROOT = CONFIG_ROOT / "baselines"
SCRIPT_ROOT = ROOT / "scripts"
BASELINE_SCRIPT_ROOT = SCRIPT_ROOT / "baselines"

OBJECTIVES = ["grad_ascent", "grad_diff", "npo", "dpo", "simnpo", "rmu", "kl", "retrain"]
WRAPPERS = ["none", "forget_only", "cadmu"]
BASELINES = [
    objective if wrapper == "none" else f"{wrapper}_{objective}"
    for wrapper in WRAPPERS
    for objective in OBJECTIVES
]

SETTINGS = {
    "strong": {"alpha_forget": 1.5, "beta_retain": 1.0},
    "weak": {"alpha_forget": 1.0, "beta_retain": 1.5},
}

TOFU_SPLITS = {
    "forget01": {
        "forget_split": "forget01",
        "retain_split": "retain99",
        "forget_perturbed_split": "forget01_perturbed",
        "holdout_split": "holdout01",
    },
    "forget05": {
        "forget_split": "forget05",
        "retain_split": "retain95",
        "forget_perturbed_split": "forget05_perturbed",
        "holdout_split": "holdout05",
    },
    "forget10": {
        "forget_split": "forget10",
        "retain_split": "retain90",
        "forget_perturbed_split": "forget10_perturbed",
        "holdout_split": "holdout10",
    },
}

# Models with per-model TOFU reference configs in the default objective sweep.
DEFAULT_SWEEP_TOFU_MODELS = [
    "deepseek_r1_distill_llama_8b",
    "deepseek_r1_distill_qwen_1_5b",
    "deepseek_r1_distill_qwen_7b",
    "gemma4_e2b",
    "llama3_3b",
    "llama3_8b",
    "ministral3_3b_instruct",
    "mistral7b_instruct_v03",
    "phi4_mini_instruct",
    "qwen3_5_0_8b",
    "qwen3_5_27b",
    "qwen3_5_2b",
    "qwen3_5_4b",
    "qwen3_5_9b",
    "qwen3_6_27b",
    "smollm3_3b",
    "zai_chatglm3_6b",
    "zai_glm_4_9b",
]

# Models with flat <model>_<dataset>_probe_relative_main_sweep.yaml references.
MUSE_FLAT_MODELS = [
    "gemma4_e2b",
    "llama3_1_8b",
    "llama3_1b",
    "llama3_2_3b",
    "ministral3_3b_instruct",
    "mistral7b_instruct_v03",
    "phi4_mini_instruct",
    "qwen3_5_0_8b",
    "qwen3_5_2b",
    "qwen3_5_4b",
    "smollm3_3b",
]

# Models whose main sweep lives in a directory tree; use the standard point.
MUSE_DIR_MODELS = ["zai_chatglm3_6b", "zai_glm_4_9b"]

DEEPSEEK_MODELS = [
    "deepseek_r1_distill_llama_8b",
    "deepseek_r1_distill_qwen_1_5b",
    "deepseek_r1_distill_qwen_7b",
]

MUSE_DATASETS = ["muse_books", "muse_news", "wmdp"]

# Data-block template source for synthesized DeepSeek muse/wmdp configs.
DEEPSEEK_DATA_TEMPLATE_MODEL = "qwen3_5_2b"

SFT_DIR_BY_DATASET = {
    "muse_books": "{model}_muse_books_qa_sft",
    "muse_news": "{model}_muse_news_qa_sft",
    "wmdp": "{model}_wmdp_sft",
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_unlearn(setting: str) -> dict:
    return {
        "baselines": list(BASELINES),
        "steps": 100,
        "batch_size": 1,
        "lr": 0.0002,
        "alpha_forget": SETTINGS[setting]["alpha_forget"],
        "beta_retain": SETTINGS[setting]["beta_retain"],
        "npo_beta": 1.0,
        "dpo_beta": 1.0,
        "simnpo_beta": 1.0,
        "rmu_target_scale": 1.0,
        "projection_eps": 1.0e-12,
        "shared_methods": [],
    }


def emit_config(model: str, tag: str, cfg: dict) -> Path:
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = 42
    cfg["save_models"] = False
    cfg["output_dir"] = f"outputs/baselines/{model}/{tag}"
    out_dir = BASELINE_ROOT / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def build_from_reference(model: str, dataset: str, setting: str, ref: dict,
                         split: str | None = None) -> dict:
    cfg = copy.deepcopy(ref)
    if split is not None:
        cfg["data"].update(TOFU_SPLITS[split])
    cfg["unlearn"] = make_unlearn(setting)
    return cfg


def tofu_reference(model: str) -> dict:
    if model in DEFAULT_SWEEP_TOFU_MODELS:
        ref = load_yaml(
            CONFIG_ROOT / "llm_model_unlearning_method_default_sweep" / f"{model}_sweep" / "grad_diff.yaml"
        )
    elif model == "llama3_1b":
        ref = load_yaml(CONFIG_ROOT / "llama3_1b_mask_compare_probe_relative.yaml")
        # Normalize the quick-comparison config to the standard baseline setup.
        ref["probe"]["activity_quantile"] = 0.1
        ref["probe"]["margin_quantile"] = 0.05
    elif model == "llama3_2_3b":
        # No TOFU assets exist for llama3_2_3b; mirror llama3_1b but point at
        # the (pending) llama3_2_3b TOFU SFT output.
        ref = load_yaml(CONFIG_ROOT / "llama3_1b_mask_compare_probe_relative.yaml")
        ref["probe"]["activity_quantile"] = 0.1
        ref["probe"]["margin_quantile"] = 0.05
        sft_path = "outputs/finetune/llama3_2_3b_tofu_sft/model"
        ref["model"]["name_or_path"] = sft_path
        ref["model"]["tokenizer_name_or_path"] = sft_path
        ref["model"]["trust_remote_code"] = False
        ref["model"]["dtype"] = "float16"
    else:
        raise ValueError(f"No TOFU reference for model: {model}")
    return ref


def muse_reference(model: str, dataset: str) -> dict:
    if model in MUSE_FLAT_MODELS:
        return load_yaml(CONFIG_ROOT / f"{model}_{dataset}_probe_relative_main_sweep.yaml")
    if model in MUSE_DIR_MODELS:
        return load_yaml(
            CONFIG_ROOT / f"{model}_{dataset}_probe_relative_main_sweep" / "margin" / "margin_q_05.yaml"
        )
    if model in DEEPSEEK_MODELS:
        ref = load_yaml(
            CONFIG_ROOT / f"{DEEPSEEK_DATA_TEMPLATE_MODEL}_{dataset}_probe_relative_main_sweep.yaml"
        )
        sft_dir = SFT_DIR_BY_DATASET[dataset].format(model=model)
        sft_path = f"outputs/finetune/{sft_dir}/model"
        ref["model"]["name_or_path"] = sft_path
        ref["model"]["tokenizer_name_or_path"] = sft_path
        ref["model"]["trust_remote_code"] = True
        ref["model"]["dtype"] = "bfloat16"
        return ref
    raise ValueError(f"No {dataset} reference for model: {model}")


def tofu_models() -> list[str]:
    return DEFAULT_SWEEP_TOFU_MODELS + ["llama3_1b", "llama3_2_3b"]


def muse_models() -> list[str]:
    return MUSE_FLAT_MODELS + MUSE_DIR_MODELS + DEEPSEEK_MODELS


def all_models() -> list[str]:
    return sorted(set(tofu_models()) | set(muse_models()))


def generate_configs() -> int:
    count = 0
    for model in tofu_models():
        ref = tofu_reference(model)
        for split in TOFU_SPLITS:
            for setting in SETTINGS:
                tag = f"tofu_{split}_{setting}"
                cfg = build_from_reference(model, "tofu", setting, ref, split=split)
                emit_config(model, tag, cfg)
                count += 1
    for model in muse_models():
        for dataset in MUSE_DATASETS:
            ref = muse_reference(model, dataset)
            for setting in SETTINGS:
                tag = f"{dataset}_{setting}"
                cfg = build_from_reference(model, dataset, setting, ref)
                emit_config(model, tag, cfg)
                count += 1
    return count


RUN_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)"
PYTHON_BIN="${{PYTHON:-python}}"
MODEL="{model}"
MODE="${{1:-all}}"

cd "$ROOT_DIR"

run_group() {{
  local group="$1"
  shopt -s nullglob
  local configs=( configs/baselines/"$MODEL"/${{group}}_*.yaml )
  shopt -u nullglob
  if [ "${{#configs[@]}}" -eq 0 ]; then
    echo "No $group baseline configs for $MODEL; skipping."
    return 0
  fi
  for cfg in "${{configs[@]}}"; do
    echo "============================================================"
    echo "Running: $cfg"
    echo "Start: $(date '+%F %T %Z')"
    "$PYTHON_BIN" scripts/run_open_tofu.py --config "$cfg"
    echo "Done:  $(date '+%F %T %Z')"
  done
}}

case "$MODE" in
  tofu|muse_books|muse_news|wmdp) run_group "$MODE" ;;
  all)
    run_group tofu
    run_group muse_books
    run_group muse_news
    run_group wmdp
    ;;
  *) echo "Usage: $0 [tofu|muse_books|muse_news|wmdp|all]" >&2; exit 1 ;;
esac
"""

MASTER_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)"
MODE="${{1:-all}}"

MODELS=(
{model_lines}
)

for model in "${{MODELS[@]}}"; do
  bash "$ROOT_DIR/scripts/baselines/run_${{model}}_baselines.sh" "$MODE"
done
"""


def generate_scripts() -> int:
    count = 0
    BASELINE_SCRIPT_ROOT.mkdir(parents=True, exist_ok=True)
    for model in all_models():
        path = BASELINE_SCRIPT_ROOT / f"run_{model}_baselines.sh"
        path.write_text(RUN_SCRIPT_TEMPLATE.format(model=model), encoding="utf-8")
        path.chmod(0o755)
        count += 1
    model_lines = "\n".join(f"  {model}" for model in all_models())
    master = BASELINE_SCRIPT_ROOT / "run_all_baselines.sh"
    master.write_text(MASTER_SCRIPT_TEMPLATE.format(model_lines=model_lines), encoding="utf-8")
    master.chmod(0o755)
    return count + 1


def validate() -> None:
    for path in sorted(BASELINE_ROOT.rglob("*.yaml")):
        cfg = load_yaml(path)
        for key in ("model", "data", "unlearn", "eval", "output_dir"):
            assert key in cfg, f"{path} missing {key}"
        assert cfg["unlearn"]["baselines"] == BASELINES, f"{path} bad baselines"
        assert cfg["unlearn"]["shared_methods"] == [], f"{path} shared_methods must be empty"
        assert "_sweep" not in cfg["output_dir"], f"{path} output_dir must avoid *_sweep"


def main() -> None:
    configs = generate_configs()
    scripts = generate_scripts()
    validate()
    print(f"wrote {configs} configs under {BASELINE_ROOT.relative_to(ROOT)}")
    print(f"wrote {scripts} run scripts under {BASELINE_SCRIPT_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
