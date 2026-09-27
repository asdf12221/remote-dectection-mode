#!/usr/bin/env bash
set -euo pipefail

# Reference-only lineage for bifpn_trainval_fp16_finaltrain_crt.pth.
# This trains on trainval, so its validation score is not an independent metric.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
CONFIG="$ROOT/configs/cascade_merged_25cls_v3_ce_v5_bifpn_trainval_fp16_finaltrain_crt_config.py"
FT_CKPT="${FT_CKPT:-$ROOT/work_dirs/cascade_merged_25cls_v3_ce_v5_bifpn_trainval_fp16_finaltrain/best_coco_bbox_mAP_epoch_24.pth}"
TRAIN_SCRIPT="$(python -c 'import mmdet, os; print(os.path.join(os.path.dirname(mmdet.__file__), ".mim/tools/train.py"))')"

torchrun --nproc_per_node="${NPROC:-1}" --master_port="${PORT:-29536}" \
  "$TRAIN_SCRIPT" "$CONFIG" --launcher pytorch \
  --cfg-options load_from="$FT_CKPT"
