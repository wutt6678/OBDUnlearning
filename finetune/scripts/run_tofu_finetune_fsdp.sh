#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 CONFIG [NPROC_PER_NODE]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="$1"
NPROC_PER_NODE="${2:-${NPROC_PER_NODE:-}}"
LOG_DIR="${FSDP_LOG_DIR:-$ROOT_DIR/outputs/finetune/fsdp_logs}"

if [ -z "$NPROC_PER_NODE" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NPROC_PER_NODE="$(python - <<'PY'
import os
visible = [item for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if item.strip()]
print(len(visible))
PY
)"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE="2"
  fi
fi

TMP_CONFIG="$(mktemp /tmp/tofu_fsdp_config_XXXXXX.yaml)"
cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

NPROC_PER_NODE_FOR_CONFIG="$NPROC_PER_NODE" "$PYTHON_BIN" - "$CONFIG" "$TMP_CONFIG" <<'PY'
import os
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg.setdefault("model", {})["device"] = "cpu"
cfg.setdefault("train", {})["distributed"] = "fsdp"
cfg["train"].setdefault("fsdp_min_num_params", 1_000_000)
cfg["train"].setdefault("fsdp_mixed_precision", cfg.get("model", {}).get("dtype", "bfloat16"))
cfg["train"].setdefault("gradient_checkpointing", False)
cfg["train"].setdefault("gradient_checkpointing_use_reentrant", False)
cfg["train"].setdefault("fsdp_cpu_offload", True)
cfg["train"].setdefault("fsdp_use_orig_params", True)
if "fsdp_batch_size_mode" not in cfg["train"]:
    batch_size = int(cfg["train"].get("batch_size", 1))
    world_size = int(os.environ.get("NPROC_PER_NODE_FOR_CONFIG", "1"))
    cfg["train"]["fsdp_batch_size_mode"] = (
        "global" if batch_size >= world_size and batch_size % world_size == 0 else "per_rank"
    )
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "Running TOFU fine-tune with FSDP"
echo "Config: $CONFIG"
echo "Temporary FSDP config: $TMP_CONFIG"
echo "nproc_per_node: $NPROC_PER_NODE"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
echo "FSDP log dir: $LOG_DIR"
echo "Start: $(date '+%F %T %Z')"
torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" --log-dir "$LOG_DIR" --tee 3 finetune/finetune_tofu.py --config "$TMP_CONFIG"
echo "Done:  $(date '+%F %T %Z')"
