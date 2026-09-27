# Cascade InternImage-L 25cls CE v5 + BiFPN neck — 高质量合成数据预训练
import os
# 基于 cascade_merged_25cls_v3_ce_config.py:
#   - neck: FPN -> BiFPN (mmdet_custom/models/necks/bifpn.py, 移植自
#     https://github.com/ViswanathaReddyGajjala/EfficientDet-Pytorch)
#   - 数据: finaldatav2 -> finaldatav5/annotations_train_synth_hq.json
#     (22766 张高质量合成图; FSC 标签已按 20260831 zip 更新)
#   - 预训练: load_from=DOTA 权重(best_coco_bbox_mAP_epoch_32.pth), 24 epochs, EMA
# 后续: 预训练完成后在真实数据(如 zipfsc)上 fine-tune, 参照
#       cascade_merged_25cls_v3_ce_v5_zipfsc_pretrain_ft12 模式
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

# --- 预训练: 加载 DOTA 训练过的权重(backbone/rpn/可复用 head 部分) ---
load_from = os.getenv('FPBA_DOTA_CKPT') or None
resume = False

# --- 高质量合成数据 (synth_hq, 22766 张) ---
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=os.getenv('FPBA_DATA_ROOT', 'data/finaldatav5'),
        ann_file='annotations_train_synth_hq.json',
        data_prefix=dict(img='images/all/'),
        filter_cfg=dict(filter_empty_gt=False)))

val_dataloader = dict(
    dataset=dict(
        data_root=os.getenv('FPBA_DATA_ROOT', 'data/finaldatav5'),
        ann_file='annotations_val.json',
        data_prefix=dict(img='images/all/')))

val_evaluator = dict(
    ann_file=os.path.join(os.getenv('FPBA_DATA_ROOT', 'data/finaldatav5'), 'annotations_val.json'))

work_dir = os.path.join(os.getenv('FPBA_WORK_ROOT', 'work_dirs'), 'cascade_merged_25cls_v3_ce_v5_bifpn_synthhq_pretrain')
