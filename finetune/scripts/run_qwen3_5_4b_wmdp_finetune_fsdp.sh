#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-$ROOT_DIR/finetune/configs/qwen3_5_4b_wmdp_sft.yaml}"
bash "$ROOT_DIR/finetune/scripts/run_tofu_finetune_fsdp.sh" "$CONFIG"
