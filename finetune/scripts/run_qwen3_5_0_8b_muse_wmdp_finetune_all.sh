#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "$ROOT_DIR/finetune/scripts/run_qwen3_5_0_8b_wmdp_finetune_fsdp.sh"
bash "$ROOT_DIR/finetune/scripts/run_qwen3_5_0_8b_muse_books_finetune_fsdp.sh"
bash "$ROOT_DIR/finetune/scripts/run_qwen3_5_0_8b_muse_news_finetune_fsdp.sh"
bash "$ROOT_DIR/finetune/scripts/run_qwen3_5_0_8b_muse_books_qa_finetune_fsdp.sh"
bash "$ROOT_DIR/finetune/scripts/run_qwen3_5_0_8b_muse_news_qa_finetune_fsdp.sh"
