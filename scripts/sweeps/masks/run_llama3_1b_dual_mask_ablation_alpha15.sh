#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="$ROOT_DIR/configs/llama3_1b_dual_mask_ablation_alpha15.yaml"

"$PYTHON_BIN" "$ROOT_DIR/scripts/run_open_tofu.py" --config "$CONFIG"
