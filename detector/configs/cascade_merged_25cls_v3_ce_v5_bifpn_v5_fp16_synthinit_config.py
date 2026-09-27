# Cascade InternImage-L 25cls CE v5 + BiFPN — backbone FP16, 加载合成预训练, v5 数据
import os
# 训练策略与 bifpn_v5_fp16 完全一致 (FP16BackboneLayerDecayOptimizerConstructor,
# frozen_stages=2, EMA, layer_decay 0.8, 24ep), 仅 load_from 换为合成预训练:
#   - load_from: synth_hq 预训练权重 (epoch_12, val mAP 0.683)
#   - 数据: annotations_train.json (v5 全量)
_base_ = ['./cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_config.py']

# --- 加载合成数据预训练模型 (BiFPN 结构一致) ---
load_from = os.getenv('FPBA_SYNTH_CKPT', 'weights/synth_pretrain_epoch12.pth')
resume = False

# --- 解冻 backbone (预训练阶段 frozen_stages=2, 这里全量训练) ---
_base_.model.backbone.frozen_stages = -1

work_dir = os.path.join(os.getenv('FPBA_WORK_ROOT', 'work_dirs'), 'cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_synthinit')
