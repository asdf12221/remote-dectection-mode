# Cascade InternImage-L 25cls CE v5 + BiFPN neck — v5 正常训练, backbone FP16
import os
# 基于 cascade_merged_25cls_v3_ce_v5_bifpn_v5_config.py:
#   - neck: FPN -> BiFPN
#   - 数据: finaldatav5 (v5 全量)
#   - backbone: 权重存 fp16 (FP16BackboneLayerDecayOptimizerConstructor 在
#     构建优化器前把 model.backbone.half();DCNv3 支持 half)
#   - QAT 第一步: 先让 backbone 以 fp16 训练, 后续 int8 fake-quant 在此基础叠加
# 注意: EMA hook 保留 (mmengine EMA 与 fp16 兼容, 交换时自动 cast);
#       若训练不稳定 (loss 异常), 回退用 bifpn_v5_config.py (纯 AMP)。
_base_ = ['./cascade_merged_25cls_v3_ce_v5_bifpn_v5_config.py']

custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet_custom.models.backbones.intern_image',
        'mmdet_custom.custom_layer_decay_optimizer_constructor',
        'mmdet_custom.models.necks.bifpn',
        'mmdet_custom.fp16_backbone_constructor',
    ])

# --- backbone fp16: 构造优化器前转 half ---
optim_wrapper = dict(
    constructor='FP16BackboneLayerDecayOptimizerConstructor')

work_dir = os.path.join(os.getenv('FPBA_WORK_ROOT', 'work_dirs'), 'cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16')
