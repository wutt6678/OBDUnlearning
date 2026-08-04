#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$ROOT_DIR/finetune/scripts/run_qwen3_5_2b_wmdp_finetune.sh"
bash "$ROOT_DIR/finetune/scripts/run_qwen3_5_2b_wmdp_eval.sh"
