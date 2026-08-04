#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="${1:-$ROOT_DIR/finetune/configs/qwen3_5_0_8b_tofu_sft_eval.yaml}"
MODEL_DIR="$ROOT_DIR/outputs/finetune/qwen3_5_0_8b_tofu_sft/model"

cd "$ROOT_DIR"
if [ ! -e "$MODEL_DIR" ]; then
  echo "Missing fine-tuned model: $MODEL_DIR" >&2
  echo "Run: bash finetune/scripts/run_qwen3_5_0_8b_tofu_finetune.sh" >&2
  exit 1
fi

echo "Running Qwen3.5-0.8B TOFU evaluation"
echo "Config: $CONFIG"
echo "Start: $(date '+%F %T %Z')"
"$PYTHON_BIN" finetune/evaluate_tofu.py --config "$CONFIG"
echo "Done:  $(date '+%F %T %Z')"
