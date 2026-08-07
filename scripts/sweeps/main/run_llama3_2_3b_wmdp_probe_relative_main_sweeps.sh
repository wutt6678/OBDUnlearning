#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$ROOT_DIR/scripts/sweeps/run_llama3_2_3b_wmdp_probe_relative_sweeps.sh" main
