#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "$ROOT_DIR/finetune/scripts/run_qwen3_6_27b_tofu_finetune.sh"
bash "$ROOT_DIR/finetune/scripts/run_qwen3_6_27b_tofu_eval.sh"
