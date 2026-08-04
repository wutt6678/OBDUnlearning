#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$ROOT_DIR/finetune/scripts/run_ministral3_3b_instruct_muse_news_finetune.sh"
bash "$ROOT_DIR/finetune/scripts/run_ministral3_3b_instruct_muse_news_eval.sh"
