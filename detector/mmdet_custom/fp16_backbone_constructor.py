#!/usr/bin/env python3
"""FP16 backbone 优化器构造器(QAT 第一步)。

在构建参数组**之前**把 model.backbone 转成 fp16:
- 必须在优化器构建前转换——若在 hook 里(优化器构建后)调 .half(),
  优化器仍持有旧的 fp32 张量引用,backbone 的梯度更新会落到无人使用的
  旧张量上,训练静默失效。
- DCNv3 CUDA 内核支持 half (AT_DISPATCH_FLOATING_TYPES_AND_HALF)。
- 用法(config):
    custom_imports = dict(imports=[..., 'mmdet_custom.fp16_backbone_constructor'])
    optim_wrapper = dict(constructor='FP16BackboneLayerDecayOptimizerConstructor', ...)
"""
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS

from .custom_layer_decay_optimizer_constructor import \
    CustomLayerDecayOptimizerConstructor


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class FP16BackboneLayerDecayOptimizerConstructor(
        CustomLayerDecayOptimizerConstructor):

    def __call__(self, model):
        if hasattr(model, 'architecture') and hasattr(model.architecture,
                                                      'backbone'):
            # mmrazor QAT 算法 (MMArchitectureQuant): 被量化检测器在
            # architecture 子模块下
            model.architecture.backbone.half()
        elif hasattr(model, 'backbone'):
            model.backbone.half()
        return super().__call__(model)
