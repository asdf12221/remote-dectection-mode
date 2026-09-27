# Reference-only trainval lineage (not used for the independent train/val metric)
import os
# 训练策略与 bifpn_v5_fp16 完全一致, 仅数据集不同:
#   - load_from: synth_hq 预训练权重 (epoch_12, val mAP 0.683)
#   - 数据: annotations_trainval.json (train+val 合并, 13394 图)
_base_ = ['./cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_config.py']

# --- 加载合成数据预训练模型 (BiFPN 结构一致) ---
load_from = os.getenv('FPBA_SYNTH_CKPT', 'weights/synth_pretrain_epoch12.pth')
resume = False

# --- 解冻 backbone (预训练阶段 frozen_stages=2, 这里全量训练) ---
_base_.model.backbone.frozen_stages = -1

# --- 数据集换成 trainval (与 v5 版唯一差异) ---
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=os.getenv('FPBA_DATA_ROOT', 'data/finaldatav5'),
        ann_file='annotations_trainval.json',
        data_prefix=dict(img='images/all/'),
        filter_cfg=dict(filter_empty_gt=False)))

work_dir = os.path.join(os.getenv('FPBA_WORK_ROOT', 'work_dirs'), 'cascade_merged_25cls_v3_ce_v5_bifpn_trainval_fp16_synthinit')
