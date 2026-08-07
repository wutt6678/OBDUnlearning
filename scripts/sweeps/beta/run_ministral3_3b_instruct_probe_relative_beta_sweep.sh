#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$ROOT_DIR/scripts/sweeps/run_ministral3_3b_instruct_probe_relative_sweeps.sh" beta
