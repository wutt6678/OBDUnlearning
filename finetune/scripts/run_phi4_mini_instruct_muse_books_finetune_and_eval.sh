#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$ROOT_DIR/finetune/scripts/run_phi4_mini_instruct_muse_books_finetune.sh"
bash "$ROOT_DIR/finetune/scripts/run_phi4_mini_instruct_muse_books_eval.sh"
