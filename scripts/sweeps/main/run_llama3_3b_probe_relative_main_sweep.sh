#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="$ROOT_DIR/configs/llama3_3b_mask_compare_probe_relative_main_sweep.yaml"

get_missing_methods() {
  "$PYTHON_BIN" - "$CONFIG" "$ROOT_DIR" <<'PYIN'
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
    for wrapper in unlearn.get(
        "wrappers", ["none", "forget_only", "cadmu"]
    )
]

completed = set()
comparison_path = out_dir / "open_tofu_comparison.json"
if comparison_path.exists():
    try:
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        completed.update(comparison.get("methods", {}))
    except Exception:
        pass
for method in methods:
    if (out_dir / method / "open_tofu_eval.json").exists():
        completed.add(method)

print("\n".join(method for method in methods if method not in completed))
PYIN
}

write_temp_config() {
  local methods_csv="$1"
  "$PYTHON_BIN" - "$CONFIG" "$methods_csv" <<'PYIN'
import os
import sys
import tempfile
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
cfg["unlearn"]["baselines"] = [
    method for method in sys.argv[2].split(",") if method
]
fd, path = tempfile.mkstemp(prefix="llama3_3b_main_sweep_", suffix=".yaml")
os.close(fd)
Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(path)
PYIN
}

missing="$(get_missing_methods | sed '/^$/d')"
if [ -z "$missing" ]; then
  echo "All Llama-3.2-3B methods are already completed."
  exit 0
fi

missing_csv="$(printf '%s,' $missing | sed 's/,$//')"
temp_config="$(write_temp_config "$missing_csv")"
trap 'rm -f "$temp_config"' EXIT

echo "============================================================"
echo "Running Llama-3.2-3B main_sweep"
echo "Missing methods: $missing_csv"
echo "Start: $(date '+%F %T %Z')"
"$PYTHON_BIN" "$ROOT_DIR/scripts/run_open_tofu.py" --config "$temp_config"
echo "Done:  $(date '+%F %T %Z')"
