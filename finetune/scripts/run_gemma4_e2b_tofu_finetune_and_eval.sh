#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "$ROOT_DIR/finetune/scripts/run_gemma4_e2b_tofu_finetune.sh"
bash "$ROOT_DIR/finetune/scripts/run_gemma4_e2b_tofu_eval.sh"
