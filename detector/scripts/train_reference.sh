#!/usr/bin/env bash
set -euo pipefail

# Reference-only lineage for the supplied trainval finaltrain checkpoint.
# Do not use this run for an independent train/val benchmark.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
CONFIG="$ROOT/configs/cascade_merged_25cls_v3_ce_v5_bifpn_trainval_fp16_finaltrain_config.py"
S1_CKPT="${S1_CKPT:-${FPBA_SYNTH_CKPT:-$ROOT/weights/synth_pretrain_epoch12.pth}}"
TRAIN_SCRIPT="$(python -c 'import mmdet, os; print(os.path.join(os.path.dirname(mmdet.__file__), ".mim/tools/train.py"))')"

torchrun --nproc_per_node="${NPROC:-1}" --master_port="${PORT:-29527}" \
  "$TRAIN_SCRIPT" "$CONFIG" --launcher pytorch \
  --cfg-options load_from="$S1_CKPT"
