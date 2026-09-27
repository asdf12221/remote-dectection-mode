# Cascade InternImage-L 25cls CE v5 + BiFPN neck — v5 正常训练(独立 work_dir)
import os
# 基于 cascade_merged_25cls_v3_ce_config.py:
#   - neck: FPN -> BiFPN
#   - 数据: finaldatav2 -> finaldatav5(12042 张)
#   - 损失: 保持 CE 原样(CrossEntropyLoss use_sigmoid=True,不用 SeesawLoss)
# BiFPN 实现: mmdet_custom/models/necks/bifpn.py
# (移植自 https://github.com/ViswanathaReddyGajjala/EfficientDet-Pytorch)
_base_ = ['./cascade_merged_25cls_v3_ce_config.py']

# --- 注册自定义 BiFPN ---
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet_custom.models.backbones.intern_image',
        'mmdet_custom.custom_layer_decay_optimizer_constructor',
        'mmdet_custom.models.necks.bifpn',
    ])

# --- FPN -> BiFPN(4 输入层 + maxpool 生成 P7,输出 5 层,stride 4..64 与 FPN 一致) ---
_base_.model.neck = dict(
    type='BiFPN',
    in_channels=[160, 320, 640, 1280],
    out_channels=256,
    norm_cfg=dict(type='SyncBN', requires_grad=True))

# --- finaldatav5 数据集 ---
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=os.getenv('FPBA_DATA_ROOT', 'data/finaldatav5'),
        ann_file='annotations_train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=os.getenv('FPBA_DATA_ROOT', 'data/finaldatav5'),
        ann_file='annotations_val.json'))

val_evaluator = dict(
    ann_file=os.path.join(os.getenv('FPBA_DATA_ROOT', 'data/finaldatav5'), 'annotations_val.json'))

work_dir = os.path.join(os.getenv('FPBA_WORK_ROOT', 'work_dirs'), 'cascade_merged_25cls_v3_ce_v5_bifpn_v5')
