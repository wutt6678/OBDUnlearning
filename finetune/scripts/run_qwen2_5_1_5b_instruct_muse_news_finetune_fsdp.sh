#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-$ROOT_DIR/finetune/configs/qwen2_5_1_5b_instruct_muse_news_sft.yaml}"
bash "$ROOT_DIR/finetune/scripts/run_tofu_finetune_fsdp.sh" "$CONFIG" "${2:-${NPROC_PER_NODE:-}}"
