#!/usr/bin/env bash
set -euo pipefail

# Stage 3 benchmark cRT: freeze the detector except the three fc_cls layers.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
CONFIG="$ROOT/configs/cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_finaltrain_abl_a0_crt_config.py"
FT_CKPT="${FT_CKPT:-$ROOT/work_dirs/cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_finaltrain_abl_a0/best_coco_bbox_mAP_epoch_35.pth}"
TRAIN_SCRIPT="$(python -c 'import mmdet, os; print(os.path.join(os.path.dirname(mmdet.__file__), ".mim/tools/train.py"))')"

torchrun --nproc_per_node="${NPROC:-1}" --master_port="${PORT:-29562}" \
  "$TRAIN_SCRIPT" "$CONFIG" --launcher pytorch \
  --cfg-options load_from="$FT_CKPT"
