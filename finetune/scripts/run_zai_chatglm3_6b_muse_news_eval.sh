#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="${1:-$ROOT_DIR/finetune/configs/zai_chatglm3_6b_muse_news_eval.yaml}"
cd "$ROOT_DIR"
echo "Running zai_chatglm3_6b MUSE-news evaluation"
echo "Config: $CONFIG"
echo "Start: $(date '+%F %T %Z')"
"$PYTHON_BIN" scripts/evaluate_benchmark.py --config "$CONFIG"
echo "Done:  $(date '+%F %T %Z')"
