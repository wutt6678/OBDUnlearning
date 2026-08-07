#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

exec "$ROOT_DIR/scripts/sweeps/beta/run_llama3_1b_probe_relative_beta_objective_sweeps.sh" grad_diff
