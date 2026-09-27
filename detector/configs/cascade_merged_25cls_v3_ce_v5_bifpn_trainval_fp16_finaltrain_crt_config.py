# Reference-only cRT lineage for the supplied trainval checkpoint.
import os
# 基于 trainval_fp16_finaltrain: CRTDetector 仅 fc_cls / RepeatFactor 均衡采样(0.15)
# / 10ep / lr 2e-3 / 无 EMA / backbone fp16 保持
_base_ = ['./cascade_merged_25cls_v3_ce_v5_bifpn_trainval_fp16_finaltrain_config.py']

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

# 1) 仅 fc_cls 可训练
_base_.model.type = 'CRTDetector'
_base_.model.unfreeze_substr = ('fc_cls', )

# 2) 均衡采样 (全类, thr 0.15; 替换 finaltrain 的 FSC-only 0.08 包装)
inner = _base_.train_dataloader.dataset.dataset if hasattr(_base_.train_dataloader.dataset, 'dataset') else _base_.train_dataloader.dataset
_base_.train_dataloader.dataset = dict(
    type='RepeatFactorDataset',
    repeat_thr=0.15,
    dataset=inner)

# 3) load_from: 由启动脚本 --cfg-options 传入 finaltrain best (此处 None 防误跑)
load_from = None
resume = False

# 4) 10ep / milestones [7,9] / lr 2e-3 (仅 fc_cls)
_base_.max_epochs = 10
_base_.train_cfg.max_epochs = 10
_base_.param_scheduler[0].end = 10
_base_.param_scheduler[0].milestones = [7, 9]
_base_.optim_wrapper.optimizer.lr = 0.002

# 5) 去掉 EMA
_base_.custom_hooks = [h for h in _base_.custom_hooks if h.get('type') != 'EMAHook']

work_dir = os.path.join(os.getenv('FPBA_WORK_ROOT', 'work_dirs'), 'cascade_merged_25cls_v3_ce_v5_bifpn_trainval_fp16_finaltrain_crt')

# 单卡 cRT: bs 2->6 (global 6 = 原 3卡x2)
train_dataloader = dict(batch_size=2)  # 3卡x2 = global 6
