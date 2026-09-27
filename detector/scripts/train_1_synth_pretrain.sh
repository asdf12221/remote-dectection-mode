#!/usr/bin/env bash
set -euo pipefail

# Stage 1: synthetic-data pretraining. Set FPBA_DATA_ROOT, FPBA_DOTA_CKPT,
# FPBA_INTERNIMAGE_CKPT and FPBA_WORK_ROOT before running.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
CONFIG="$ROOT/configs/cascade_merged_25cls_v3_ce_v5_bifpn_synthhq_pretrain_config.py"
TRAIN_SCRIPT="$(python -c 'import mmdet, os; print(os.path.join(os.path.dirname(mmdet.__file__), ".mim/tools/train.py"))')"

torchrun --nproc_per_node="${NPROC:-1}" --master_port="${PORT:-29560}" \
  "$TRAIN_SCRIPT" "$CONFIG" --launcher pytorch
