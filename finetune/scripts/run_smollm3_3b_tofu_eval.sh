#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="${1:-$ROOT_DIR/finetune/configs/smollm3_3b_tofu_sft_eval.yaml}"
MODEL_DIR="$ROOT_DIR/outputs/finetune/smollm3_3b_tofu_sft/model"

cd "$ROOT_DIR"
if [ ! -e "$MODEL_DIR" ]; then
  echo "Missing fine-tuned model: $MODEL_DIR" >&2
  echo "Run: bash finetune/scripts/run_smollm3_3b_tofu_finetune.sh" >&2
  exit 1
fi

echo "Running SmolLM3-3B TOFU evaluation"
echo "Config: $CONFIG"
echo "Start: $(date '+%F %T %Z')"
"$PYTHON_BIN" finetune/evaluate_tofu.py --config "$CONFIG"
echo "Done:  $(date '+%F %T %Z')"
