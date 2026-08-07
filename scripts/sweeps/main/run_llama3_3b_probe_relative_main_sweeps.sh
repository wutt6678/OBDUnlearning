#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODE="${1:-all}"

case "$MODE" in
  activity)
    exec bash "$ROOT_DIR/scripts/sweeps/run_llama3_3b_probe_relative_sweeps.sh" activity
    ;;
  margin)
    exec bash "$ROOT_DIR/scripts/sweeps/run_llama3_3b_probe_relative_sweeps.sh" margin
    ;;
  all)
    exec bash "$ROOT_DIR/scripts/sweeps/run_llama3_3b_probe_relative_sweeps.sh" stronger
    ;;
  *)
    echo "Usage: $0 [activity|margin|all]" >&2
    exit 1
    ;;
esac
