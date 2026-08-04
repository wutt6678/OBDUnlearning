#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
MODE="${1:-all}"

get_missing_methods() {
  local cfg="$1"
  "$PYTHON_BIN" - "$cfg" "$ROOT_DIR" <<'PYIN'
import json
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
root = Path(sys.argv[2])
obj = yaml.safe_load(cfg_path.read_text())
out_dir = root / obj["output_dir"]
unlearn_cfg = obj.get("unlearn", {})
if "baselines" in unlearn_cfg:
    methods = list(unlearn_cfg["baselines"])
else:
    objective = unlearn_cfg.get("objective", "grad_diff")
    wrappers = unlearn_cfg.get("wrappers", ["none", "forget_only", "cadmu"])
    methods = [objective if wrapper in ("none", None) else f"{wrapper}_{objective}" for wrapper in wrappers]
comparison_path = out_dir / "open_tofu_comparison.json"

completed = set()
if comparison_path.exists():
    try:
        comparison = json.loads(comparison_path.read_text())
        completed = set(comparison.get("methods", {}).keys())
    except Exception:
        completed = set()

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

missing = [method for method in methods if method not in completed]
print("\n".join(missing))
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

cfg_path = Path(sys.argv[1])
methods_csv = sys.argv[2]
obj = yaml.safe_load(cfg_path.read_text())
obj["unlearn"]["baselines"] = [m for m in methods_csv.split(",") if m]
fd, temp_path = tempfile.mkstemp(prefix="llama3_1b_split_alpha_sweep_", suffix=".yaml")
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

    echo "============================================================"
    echo "Running: $cfg"
    echo "Missing methods: $missing_csv"
    echo "Start: $(date '+%F %T %Z')"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/run_open_tofu.py" --config "$temp_cfg"
    echo "Done:   $(date '+%F %T %Z')"
    rm -f "$temp_cfg"
  done
}

case "$MODE" in
  forget01|01)
    run_pattern "configs/llama3_1b_mask_compare_probe_relative_split_alpha_sweep/forget01_alpha_sweep/alpha/*.yaml"
    ;;
  forget05|05)
    run_pattern "configs/llama3_1b_mask_compare_probe_relative_split_alpha_sweep/forget05_alpha_sweep/alpha/*.yaml"
    ;;
  all)
    run_pattern "configs/llama3_1b_mask_compare_probe_relative_split_alpha_sweep/forget01_alpha_sweep/alpha/*.yaml"
    run_pattern "configs/llama3_1b_mask_compare_probe_relative_split_alpha_sweep/forget05_alpha_sweep/alpha/*.yaml"
    ;;
  *)
    echo "Usage: $0 [forget01|forget05|all]" >&2
    exit 1
    ;;
esac
