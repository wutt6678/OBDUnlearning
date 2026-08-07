#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
MODE="${1:-all}"

get_missing_methods() {
  local cfg="$1"
  "$PYTHON_BIN" - "$cfg" "$ROOT_DIR" <<'PYIN'
import json
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_dir = Path(sys.argv[2]) / cfg["output_dir"]
unlearn = cfg["unlearn"]
if "baselines" in unlearn:
    methods = list(unlearn["baselines"])
else:
    objective = unlearn.get("objective", "grad_diff")
    methods = [
        objective if wrapper in ("none", None) else f"{wrapper}_{objective}"
        for wrapper in unlearn.get("wrappers", ["none", "cadmu"])
    ]

completed = set()
comparison_path = out_dir / "open_tofu_comparison.json"
if comparison_path.exists():
    try:
        completed.update(
            json.loads(comparison_path.read_text(encoding="utf-8"))
            .get("methods", {})
        )
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
  "$PYTHON_BIN" - "$cfg" "$methods_csv" "$ROOT_DIR" <<'PYIN'
import os
import sys
import tempfile
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[3])
for key in ("name_or_path", "tokenizer_name_or_path"):
    value = cfg.get("model", {}).get(key)
    if isinstance(value, str) and value.startswith("outputs/"):
        cfg["model"][key] = str((root / value).resolve())
methods = [method for method in sys.argv[2].split(",") if method]
for shared in cfg["unlearn"].get("shared_methods", []):
    if shared not in methods:
        methods.insert(0, shared)
cfg["unlearn"]["baselines"] = methods
fd, path = tempfile.mkstemp(prefix="qwen3_5_2b_sweep_", suffix=".yaml")
os.close(fd)
Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(path)
PYIN
}

run_pattern() {
  local pattern="$1"
  shopt -s nullglob
  local configs=( $ROOT_DIR/$pattern )
  shopt -u nullglob
  if [ "${#configs[@]}" -eq 0 ]; then
    echo "No configs matched: $pattern" >&2
    exit 1
  fi

  for cfg in "${configs[@]}"; do
    local missing missing_csv temp_cfg=""
    missing="$(get_missing_methods "$cfg" | sed '/^$/d')"
    if [ -z "$missing" ]; then
      echo "Skipping completed: $cfg"
      continue
    fi
    missing_csv="$(printf '%s,' $missing | sed 's/,$//')"
    temp_cfg="$(write_temp_config "$cfg" "$missing_csv")"
    trap 'if [ -n "${temp_cfg:-}" ]; then rm -f "$temp_cfg"; fi' EXIT
    echo "============================================================"
    echo "Running: $cfg"
    echo "Missing methods: $missing_csv"
    echo "Start: $(date '+%F %T %Z')"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/run_open_tofu.py" --config "$temp_cfg"
    echo "Done:  $(date '+%F %T %Z')"
    rm -f "$temp_cfg"
    trap - EXIT
  done
}

case "$MODE" in
  alpha)
    run_pattern "configs/qwen3_5_2b_mask_compare_probe_relative_alpha_sweep/alpha/*.yaml"
    ;;
  beta)
    run_pattern "configs/qwen3_5_2b_mask_compare_probe_relative_beta_sweep/beta/*.yaml"
    ;;
  activity)
    run_pattern "configs/qwen3_5_2b_mask_compare_probe_relative_main_sweep/activity/*.yaml"
    ;;
  margin)
    run_pattern "configs/qwen3_5_2b_mask_compare_probe_relative_main_sweep/margin/*.yaml"
    ;;
  stronger)
    run_pattern "configs/qwen3_5_2b_mask_compare_probe_relative_main_sweep/activity/*.yaml"
    run_pattern "configs/qwen3_5_2b_mask_compare_probe_relative_main_sweep/margin/*.yaml"
    ;;
  all)
    run_pattern "configs/qwen3_5_2b_mask_compare_probe_relative_alpha_sweep/alpha/*.yaml"
    run_pattern "configs/qwen3_5_2b_mask_compare_probe_relative_beta_sweep/beta/*.yaml"
    run_pattern "configs/qwen3_5_2b_mask_compare_probe_relative_main_sweep/activity/*.yaml"
    run_pattern "configs/qwen3_5_2b_mask_compare_probe_relative_main_sweep/margin/*.yaml"
    ;;
  *)
    echo "Usage: $0 [alpha|beta|activity|margin|stronger|all]" >&2
    exit 1
    ;;
esac
