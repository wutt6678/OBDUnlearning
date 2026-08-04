#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG_DIR="$ROOT_DIR/configs/llama3_1b_mask_construction_usage_compare"

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
objective = unlearn.get("objective", "grad_diff")
methods = [
    objective if wrapper in ("none", None) else f"{wrapper}_{objective}"
    for wrapper in unlearn.get("wrappers", [])
]
completed = set()
comparison_path = out_dir / "open_tofu_comparison.json"
if comparison_path.exists():
    try:
        completed.update(json.loads(comparison_path.read_text(encoding="utf-8")).get("methods", {}))
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
for shared in cfg["unlearn"].get("shared_methods", []):
    if shared not in methods:
        methods.insert(0, shared)
cfg["unlearn"]["baselines"] = methods
fd, path = tempfile.mkstemp(prefix="llama3_1b_mask_compare_", suffix=".yaml")
os.close(fd)
Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(path)
PYIN
}

shopt -s nullglob
configs=( "$CONFIG_DIR"/*.yaml )
shopt -u nullglob
if [ "${#configs[@]}" -eq 0 ]; then
  echo "No configs found. Run: python scripts/materialize_llama3_1b_mask_construction_usage_compare.py" >&2
  exit 1
fi

for cfg in "${configs[@]}"; do
  missing="$(get_missing_methods "$cfg" | sed '/^$/d')"
  if [ -z "$missing" ]; then
    echo "============================================================"
    echo "Skipping completed: $cfg"
    continue
  fi
  missing_csv="$(printf '%s,' $missing | sed 's/,$//')"
  temp_cfg="$(write_temp_config "$cfg" "$missing_csv")"
  trap 'rm -f "$temp_cfg"' EXIT
  echo "============================================================"
  echo "Running: $cfg"
  echo "Missing methods: $missing_csv"
  echo "Start: $(date '+%F %T %Z')"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/run_open_tofu.py" --config "$temp_cfg"
  echo "Done:  $(date '+%F %T %Z')"
  rm -f "$temp_cfg"
  trap - EXIT
done
