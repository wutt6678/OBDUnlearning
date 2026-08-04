#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="${1:-$ROOT_DIR/finetune/configs/llama3_1b_muse_news_sft.yaml}"
cd "$ROOT_DIR"
echo "Running Llama3.2-1B muse news fine-tune"
echo "Config: $CONFIG"
echo "Start: $(date '+%F %T %Z')"
"$PYTHON_BIN" finetune/finetune_tofu.py --config "$CONFIG"
echo "Done:  $(date '+%F %T %Z')"
