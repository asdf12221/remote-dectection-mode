#!/usr/bin/env python3
"""FP32SyncBN: 强制 fp32 计算的 SyncBN。

问题: AmpOptimWrapper 的 autocast 下, SyncBN 的 batch 统计量以 fp16 计算,
特征值较大时 sumsq 溢出 fp16(>65504) / 精度丢失, running_var 被污染
(可为负值), eval 模式归一化失真 → 推理检测数暴跌甚至为 0。

修复: forward 里把输入提升到 fp32 再走原生 SyncBN,统计量全程 fp32。

用法(config, bbox_head 的 norm_cfg):
    norm_cfg=dict(type='FP32SyncBN', requires_grad=True)
"""
import torch
from mmcv.cnn.bricks.norm import MODELS, SyncBatchNorm


@MODELS.register_module()
class FP32SyncBN(SyncBatchNorm):

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        return super().forward(x)
