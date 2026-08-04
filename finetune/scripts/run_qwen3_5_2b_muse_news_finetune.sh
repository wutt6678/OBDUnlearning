#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="${1:-$ROOT_DIR/finetune/configs/qwen3_5_2b_muse_news_sft.yaml}"
cd "$ROOT_DIR"
echo "Running Qwen3.5-2B muse news fine-tune"
echo "Config: $CONFIG"
echo "Start: $(date '+%F %T %Z')"
"$PYTHON_BIN" finetune/finetune_tofu.py --config "$CONFIG"
echo "Done:  $(date '+%F %T %Z')"
