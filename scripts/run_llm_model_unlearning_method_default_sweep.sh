#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
MODEL="${1:-all}"
OBJECTIVE="${2:-all}"
OBJECTIVES=(grad_diff grad_ascent npo dpo simnpo rmu kl retrain)

get_missing_methods() {
  local cfg="$1"
  "$PYTHON_BIN" - "$cfg" "$ROOT_DIR" <<'PYIN'
import json
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
root = Path(sys.argv[2])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
out_dir = root / cfg["output_dir"]
unlearn_cfg = cfg.get("unlearn", {})

if "baselines" in unlearn_cfg:
    methods = list(unlearn_cfg["baselines"])
else:
    objective = unlearn_cfg.get("objective", "grad_diff")
    wrappers = unlearn_cfg.get("wrappers", ["none", "forget_only", "cadmu"])
    methods = [
        objective if wrapper in ("none", None) else f"{wrapper}_{objective}"
        for wrapper in wrappers
    ]

completed = set()
comparison_path = out_dir / "open_tofu_comparison.json"
if comparison_path.exists():
    try:
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        completed.update(comparison.get("methods", {}).keys())
    except Exception:
        pass

for method in methods:
    if (out_dir / method / "open_tofu_eval.json").exists():
        completed.add(method)

print("\n".join(method for method in methods if method not in completed))
PYIN
}

write_temp_config() {
  local cfg="$1"
  local methods_csv="$2"
  "$PYTHON_BIN" - "$cfg" "$methods_csv" <<'PYIN'
import os
import sys
import tempfile
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
methods = [method for method in sys.argv[2].split(",") if method]
cfg["unlearn"]["baselines"] = methods
fd, temp_path = tempfile.mkstemp(prefix="llm_model_unlearning_method_default_", suffix=".yaml")
os.close(fd)
Path(temp_path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(temp_path)
PYIN
}

run_config() {
  local cfg="$1"
  local missing missing_csv temp_cfg
  missing="$(get_missing_methods "$cfg" | sed '/^$/d')"
  if [ -z "$missing" ]; then
    echo "============================================================"
    echo "Skipping completed: $cfg"
    return
  fi
  missing_csv="$(printf '%s,' $missing | sed 's/,$//')"
  temp_cfg="$(write_temp_config "$cfg" "$missing_csv")"
  trap 'rm -f "$temp_cfg"' EXIT
  echo "============================================================"
  echo "Running: $cfg"
  echo "Missing methods: $missing_csv"
  echo "Start: $(date '+%F %T %Z')"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/run_open_tofu.py" --config "$temp_cfg"
  echo "Done:   $(date '+%F %T %Z')"
  rm -f "$temp_cfg"
  trap - EXIT
}

run_pattern() {
  local pattern="$1"
  shopt -s nullglob
  local configs=( $pattern )
  shopt -u nullglob
  if [ "${#configs[@]}" -eq 0 ]; then
    echo "No configs matched: $pattern" >&2
    exit 1
  fi
  for cfg in "${configs[@]}"; do
    run_config "$cfg"
  done
}

"$PYTHON_BIN" "$ROOT_DIR/scripts/materialize_llm_model_unlearning_method_default_sweep.py"

CONFIG_ROOT="$ROOT_DIR/configs/llm_model_unlearning_method_default_sweep"
if [ "$MODEL" = "all" ] && [ "$OBJECTIVE" = "all" ]; then
  run_pattern "$CONFIG_ROOT"/*_sweep/*.yaml
elif [ "$MODEL" = "all" ]; then
  case "$OBJECTIVE" in
    grad_diff|grad_ascent|npo|dpo|simnpo|rmu|kl|retrain)
      run_pattern "$CONFIG_ROOT"/*_sweep/"$OBJECTIVE".yaml
      ;;
    *)
      echo "Usage: $0 [all|model_slug] [all|grad_diff|grad_ascent|npo|dpo|simnpo|rmu|kl|retrain]" >&2
      exit 1
      ;;
  esac
elif [ "$OBJECTIVE" = "all" ]; then
  run_pattern "$CONFIG_ROOT/${MODEL}_sweep"/*.yaml
else
  case "$OBJECTIVE" in
    grad_diff|grad_ascent|npo|dpo|simnpo|rmu|kl|retrain)
      run_config "$CONFIG_ROOT/${MODEL}_sweep/${OBJECTIVE}.yaml"
      ;;
    *)
      echo "Usage: $0 [all|model_slug] [all|grad_diff|grad_ascent|npo|dpo|simnpo|rmu|kl|retrain]" >&2
      exit 1
      ;;
  esac
fi
