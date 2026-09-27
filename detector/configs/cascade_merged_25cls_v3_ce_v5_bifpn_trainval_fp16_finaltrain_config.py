# Reference-only trainval finaltrain (not used for the independent train/val metric)
import os
# = trainval_fp16_synthinit_config.py 的副本, 仅 work_dir 不同 (finaltrain)
#   (annotations_trainval.json 已原地更新为 0902 完整重标 FSC 标签)
# 配方: 合成预训练 epoch_12 + backbone 解冻(-1) + fp16 + 24ep
#   + RepeatFactorDataset 仅 FSC 温和重采样 (repeat_thr=0.08 → FSC ≈ 1.45×)
_base_ = ['./cascade_merged_25cls_v3_ce_v5_bifpn_trainval_fp16_synthinit_config.py']

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

work_dir = os.path.join(os.getenv('FPBA_WORK_ROOT', 'work_dirs'), 'cascade_merged_25cls_v3_ce_v5_bifpn_trainval_fp16_finaltrain')
