#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="${1:-$ROOT_DIR/finetune/configs/mistral7b_instruct_v03_muse_news_qa_eval.yaml}"
MODEL_DIR="$ROOT_DIR/outputs/finetune/mistral7b_instruct_v03_muse_news_qa_sft/model"
if [ ! -e "$MODEL_DIR" ]; then
  echo "Missing model directory: $MODEL_DIR" >&2
  echo "Run the matching fine-tune script first." >&2
  exit 1
fi
cd "$ROOT_DIR"
echo "Running Mistral-7B-Instruct-v0.3 MUSE-News QA evaluation"
echo "Config: $CONFIG"
echo "Start: $(date '+%F %T %Z')"
"$PYTHON_BIN" scripts/evaluate_benchmark.py --config "$CONFIG"
echo "Done:  $(date '+%F %T %Z')"
