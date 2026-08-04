#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="${1:-$ROOT_DIR/finetune/configs/deepseek_r1_distill_qwen_1_5b_tofu_sft_eval.yaml}"
cd "$ROOT_DIR"
echo "Running deepseek_r1_distill_qwen_1_5b TOFU evaluation"
echo "Config: $CONFIG"
echo "Start: $(date '+%F %T %Z')"
"$PYTHON_BIN" finetune/evaluate_tofu.py --config "$CONFIG"
echo "Done:  $(date '+%F %T %Z')"
