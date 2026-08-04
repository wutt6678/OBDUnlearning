#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
OBJECTIVE="${1:-all}"
MODE="${2:-all}"
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
obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
out_dir = root / obj["output_dir"]
unlearn_cfg = obj.get("unlearn", {})
if "baselines" in unlearn_cfg:
    methods = list(unlearn_cfg["baselines"])
else:
    objective = unlearn_cfg.get("objective", "grad_diff")
    wrappers = unlearn_cfg.get("wrappers", ["none", "forget_only", "cadmu"])
    methods = [objective if wrapper in ("none", None) else f"{wrapper}_{objective}" for wrapper in wrappers]

completed = set()
comparison_path = out_dir / "open_tofu_comparison.json"
if comparison_path.exists():
    try:
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        completed.update(comparison.get("methods", {}).keys())
    except Exception:
        pass

shared_methods = set(unlearn_cfg.get("shared_methods", []))
shared_root = None
for path in (out_dir, *out_dir.parents):
    if path.name.endswith("_sweep"):
        shared_root = path / "shared"
        break
for method in methods:
    if (out_dir / method / "open_tofu_eval.json").exists():
        completed.add(method)
    if shared_root is not None and method in shared_methods and (shared_root / method / "open_tofu_eval.json").exists():
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

obj = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
methods = [method for method in sys.argv[2].split(",") if method]
for shared_method in obj["unlearn"].get("shared_methods", []):
    if shared_method not in methods:
        methods.insert(0, shared_method)
obj["unlearn"]["baselines"] = methods
fd, temp_path = tempfile.mkstemp(prefix="llama3_1b_main_objective_sweep_", suffix=".yaml")
os.close(fd)
Path(temp_path).write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")
print(temp_path)
PYIN
}

run_pattern() {
  local pattern="$1"
  shopt -s nullglob
  local glob_pattern="$ROOT_DIR/$pattern"
  local configs=( $glob_pattern )
  shopt -u nullglob
  if [ "${#configs[@]}" -eq 0 ]; then
    echo "No configs matched: $pattern" >&2
    exit 1
  fi
  for cfg in "${configs[@]}"; do
    local missing
    missing="$(get_missing_methods "$cfg" | sed '/^$/d')"
    if [ -z "$missing" ]; then
      echo "============================================================"
      echo "Skipping completed: $cfg"
      continue
    fi
    local missing_csv temp_cfg
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
  done
}

run_objective() {
  local objective="$1"
  local mode="$2"
  case "$mode" in
    activity)
      run_pattern "configs/llama3_1b_mask_compare_probe_relative_main_objective_sweeps/${objective}_main_sweep/activity/*.yaml"
      ;;
    margin)
      run_pattern "configs/llama3_1b_mask_compare_probe_relative_main_objective_sweeps/${objective}_main_sweep/margin/*.yaml"
      ;;
    all)
      run_pattern "configs/llama3_1b_mask_compare_probe_relative_main_objective_sweeps/${objective}_main_sweep/activity/*.yaml"
      run_pattern "configs/llama3_1b_mask_compare_probe_relative_main_objective_sweeps/${objective}_main_sweep/margin/*.yaml"
      ;;
    *)
      echo "Usage: $0 [all|grad_diff|grad_ascent|npo|dpo|simnpo|rmu|kl|retrain] [activity|margin|all]" >&2
      exit 1
      ;;
  esac
}

"$PYTHON_BIN" "$ROOT_DIR/scripts/materialize_llama3_1b_probe_relative_main_objective_sweeps.py"
if [ "$OBJECTIVE" = "all" ]; then
  for objective in "${OBJECTIVES[@]}"; do
    run_objective "$objective" "$MODE"
  done
else
  case "$OBJECTIVE" in
    grad_diff|grad_ascent|npo|dpo|simnpo|rmu|kl|retrain)
      run_objective "$OBJECTIVE" "$MODE"
      ;;
    *)
      echo "Usage: $0 [all|grad_diff|grad_ascent|npo|dpo|simnpo|rmu|kl|retrain] [activity|margin|all]" >&2
      exit 1
      ;;
  esac
fi
