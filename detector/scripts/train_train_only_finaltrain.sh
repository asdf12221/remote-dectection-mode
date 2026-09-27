#!/usr/bin/env bash
set -euo pipefail

# Stage 2 benchmark training: train-only split, evaluated on the untouched val split.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
CONFIG="$ROOT/configs/cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_finaltrain_abl_a0_config.py"
S1_CKPT="${S1_CKPT:-${FPBA_SYNTH_CKPT:-$ROOT/weights/synth_pretrain_epoch12.pth}}"
TRAIN_SCRIPT="$(python -c 'import mmdet, os; print(os.path.join(os.path.dirname(mmdet.__file__), ".mim/tools/train.py"))')"

torchrun --nproc_per_node="${NPROC:-1}" --master_port="${PORT:-29561}" \
  "$TRAIN_SCRIPT" "$CONFIG" --launcher pytorch \
  --cfg-options load_from="$S1_CKPT"
