# -*- coding: utf-8 -*-
import os
# ===== C1 最终模型: A0 完整配方 best + cRT 分类器重训 =====
# 与 v5_fp16_finaltrain_crt 模板同构; 挂在 abl_a0(消融数据集)上
# load_from: A0 best (启动脚本 --cfg-options 注入); val 沿用 v5 val
_base_ = ['./cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_finaltrain_abl_a0_config.py']

custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet_custom.models.backbones.intern_image',
        'mmdet_custom.custom_layer_decay_optimizer_constructor',
        'mmdet_custom.models.necks.bifpn',
        'mmdet_custom.fp16_backbone_constructor',
        'mmdet_custom.datasets.repeat_factor_dataset',
        'mmdet_custom.models.detectors.crt_detector',
    ])

# 1) CRTDetector: 仅 fc_cls 可训练 (backbone/BiFPN/RPN/回归 全冻结)
_base_.model.type = 'CRTDetector'
_base_.model.unfreeze_substr = ('fc_cls', )

# 2) 类别均衡重采样: 换包为全类 thr 0.15 (替换主阶段 FSC-only 0.08 包装)
inner = _base_.train_dataloader.dataset.dataset if hasattr(_base_.train_dataloader.dataset, 'dataset') else _base_.train_dataloader.dataset
_base_.train_dataloader.dataset = dict(
    type='RepeatFactorDataset',
    repeat_thr=0.15,
    dataset=inner)

# 3) load_from: 由启动脚本 --cfg-options 注入本臂 best (此处 None 防误跑)
load_from = None
resume = False

# 4) 10ep / milestones [7,9] / lr 2e-3
_base_.max_epochs = 10
_base_.train_cfg.max_epochs = 10
_base_.param_scheduler[0].end = 10
_base_.param_scheduler[0].milestones = [7, 9]
_base_.optim_wrapper.optimizer.lr = 0.002

# 5) 去掉 EMA
_base_.custom_hooks = [h for h in _base_.custom_hooks if h.get('type') != 'EMAHook']

work_dir = os.path.join(os.getenv('FPBA_WORK_ROOT', 'work_dirs'), 'cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_finaltrain_abl_a0_crt')

# 单卡 cRT: bs 2->6 (global 6 = 原 3卡x2)
train_dataloader = dict(batch_size=6)
