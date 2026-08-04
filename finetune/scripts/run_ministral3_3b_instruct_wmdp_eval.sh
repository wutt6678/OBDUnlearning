#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="${1:-$ROOT_DIR/finetune/configs/ministral3_3b_instruct_wmdp_eval.yaml}"
MODEL_DIR="$ROOT_DIR/outputs/finetune/ministral3_3b_instruct_wmdp_sft/model"
cd "$ROOT_DIR"
if [ ! -e "$MODEL_DIR" ]; then
  echo "Missing fine-tuned model: $MODEL_DIR" >&2
  echo "Run: bash finetune/scripts/run_ministral3_3b_instruct_wmdp_finetune.sh" >&2
  exit 1
fi
echo "Running Ministral3-3B-Instruct wmdp evaluation"
echo "Config: $CONFIG"
echo "Start: $(date '+%F %T %Z')"
"$PYTHON_BIN" scripts/evaluate_benchmark.py --config "$CONFIG"
echo "Done:  $(date '+%F %T %Z')"
