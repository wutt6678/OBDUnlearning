#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="${1:-$ROOT_DIR/finetune/configs/ministral3_3b_instruct_muse_news_eval.yaml}"
MODEL_DIR="$ROOT_DIR/outputs/finetune/ministral3_3b_instruct_muse_news_sft/model"
cd "$ROOT_DIR"
if [ ! -e "$MODEL_DIR" ]; then
  echo "Missing fine-tuned model: $MODEL_DIR" >&2
  echo "Run: bash finetune/scripts/run_ministral3_3b_instruct_muse_news_finetune.sh" >&2
  exit 1
fi
echo "Running Ministral3-3B-Instruct muse news evaluation"
echo "Config: $CONFIG"
echo "Start: $(date '+%F %T %Z')"
"$PYTHON_BIN" scripts/evaluate_benchmark.py --config "$CONFIG"
echo "Done:  $(date '+%F %T %Z')"
