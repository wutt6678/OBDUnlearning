#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$ROOT_DIR/scripts/sweeps/run_qwen3_5_4b_muse_books_probe_relative_sweeps.sh" alpha
