# BiFPN fp16 解冻版 (fp16_synthinit 原配方) — train-only benchmark split
import os
# = v5_fp16_synthinit_config.py 的副本, 仅 work_dir 不同 (finaltrain)
#   (annotations_train.json 已原地更新为 0902 完整重标 FSC 标签)
# 配方: 合成预训练 epoch_12 + backbone 解冻(-1) + fp16 + 24ep
#   + RepeatFactorDataset 仅 FSC 温和重采样 (repeat_thr=0.08 → FSC ≈ 1.45×)
_base_ = ['./cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_synthinit_config.py']

custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet_custom.models.backbones.intern_image',
        'mmdet_custom.custom_layer_decay_optimizer_constructor',
        'mmdet_custom.models.necks.bifpn',
        'mmdet_custom.fp16_backbone_constructor',
        'mmdet_custom.datasets.repeat_factor_dataset',
    ])

# --- 仅 FSC 温和重采样 (飞机/舰船不重采样) ---
_base_.train_dataloader.dataset = dict(
    type='RepeatFactorDataset',
    repeat_thr=0.08,
    resample_classes=[24],
    dataset=_base_.train_dataloader.dataset)


# 36ep: milestones 24/30 (24ep 的 16/20 等比放大)
_base_.max_epochs = 36
_base_.train_cfg.max_epochs = 36
_base_.param_scheduler[0].end = 36
_base_.param_scheduler[0].milestones = [24, 30]
work_dir = os.path.join(os.getenv('FPBA_WORK_ROOT', 'work_dirs'), 'cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_finaltrain_abl_a0')

# === A0 基线: 完整配方 + 消融数据集 (官方 zip 3138, v5 同名标签) ===
# 与 A1-A5 唯一区别: 无消融 (保留 FSC 重采样 0.08 / fp16 / 解冻 / 合成预训练 / BiFPN)
_inner_ds = (_base_.train_dataloader.dataset.dataset
             if _base_.train_dataloader.dataset.type == 'RepeatFactorDataset'
             else _base_.train_dataloader.dataset)
_inner_ds.data_root = os.getenv('FPBA_TRAIN_ONLY_ROOT', 'data/train_only')
_inner_ds.ann_file = os.getenv('FPBA_TRAIN_ONLY_ANN', 'annotations_train_abl.json')
_inner_ds.data_prefix = dict(img=os.getenv('FPBA_TRAIN_IMAGE_PREFIX', 'images/train/'))

# 单卡消融: bs 2->6 (1卡x6 = 原3卡x2 的总batch)
train_dataloader = dict(batch_size=6)
