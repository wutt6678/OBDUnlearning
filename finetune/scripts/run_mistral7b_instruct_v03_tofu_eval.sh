#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="${1:-$ROOT_DIR/finetune/configs/mistral7b_instruct_v03_tofu_sft_eval.yaml}"
MODEL_DIR="$ROOT_DIR/outputs/finetune/mistral7b_instruct_v03_tofu_sft/model"

cd "$ROOT_DIR"
if [ ! -e "$MODEL_DIR" ]; then
  echo "Missing fine-tuned model: $MODEL_DIR" >&2
  echo "Run: bash finetune/scripts/run_mistral7b_instruct_v03_tofu_finetune.sh" >&2
  exit 1
fi

echo "Running Mistral-7B-Instruct-v0.3 TOFU evaluation"
echo "Config: $CONFIG"
echo "Start: $(date '+%F %T %Z')"
"$PYTHON_BIN" finetune/evaluate_tofu.py --config "$CONFIG"
echo "Done:  $(date '+%F %T %Z')"
