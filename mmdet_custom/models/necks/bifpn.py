# 移植自 https://github.com/ViswanathaReddyGajjala/EfficientDet-Pytorch 的 BiFPN,
# 保持原拓扑(5 层,top-down + bottom-up 加权融合,nearest 上采样 / maxpool 下采样,
# 融合后 depthwise 3x3 conv -> ReLU -> BN),修复了原实现的 bug:
#   1. 融合权重改为 nn.Parameter(原实现是普通 tensor:不参与优化、不进 checkpoint、不上 GPU)
#   2. 补上缺失的 resize:P6_td 融合前对 P7_td 上采样、P6_out 融合前对 P5_out 下采样、
#      P7_out 融合前对 P6_out 下采样(原 demo 各层同分辨率所以没暴露)
#   3. 修正 P5_out 归一化分母漏 w1 的问题
#   4. 输出通道参数化(out_channels,默认 256),输入为 4 层时用 maxpool 生成 P7
import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmdet.utils import OptConfigType
from mmcv.cnn import build_norm_layer


@MODELS.register_module()
class BiFPN(BaseModule):
    """EfficientDet 风格加权双向 FPN(移植自 ViswanathaReddyGajjala/EfficientDet-Pytorch).

    Args:
        in_channels (Sequence[int]): backbone 各层输出通道 [P3, P4, P5, P6]。
        out_channels (int): BiFPN 输出通道,须与 RPN / ROI head 的 in_channels 一致。
        norm_cfg (dict): 归一化层配置(bs 小建议 SyncBN,与 head 一致)。
        init_cfg (dict | None): 初始化配置。
    """

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=None):
        super(BiFPN, self).__init__(init_cfg=init_cfg)
        P3_channels, P4_channels, P5_channels, P6_channels = in_channels
        self.W = out_channels

        # ---- top-down path 的 3x3 卷积(原始通道 -> W)----
        self.p6_td_conv = nn.Conv2d(P6_channels, self.W, kernel_size=3,
                                    stride=1, bias=True, padding=1)
        self.p5_td_conv = nn.Conv2d(P5_channels, self.W, kernel_size=3,
                                    stride=1, bias=True, padding=1)
        self.p4_td_conv = nn.Conv2d(P4_channels, self.W, kernel_size=3,
                                    stride=1, bias=True, padding=1)
        self.p3_out_conv = nn.Conv2d(P3_channels, self.W, kernel_size=3,
                                     stride=1, bias=True, padding=1)
        self.p7_td_conv = nn.Conv2d(P6_channels, self.W, kernel_size=3,
                                    stride=1, bias=True, padding=1)

        # ---- 融合节点后的 depthwise 3x3 conv -> ReLU -> BN ----
        def _dw_node():
            return nn.Sequential(
                nn.Conv2d(self.W, self.W, kernel_size=3, stride=1,
                          groups=self.W, bias=True, padding=1),
                nn.ReLU(),
                build_norm_layer(norm_cfg, self.W)[1])

        self.p6_td_node = _dw_node()
        self.p5_td_node = _dw_node()
        self.p4_td_node = _dw_node()
        self.p3_out_node = _dw_node()
        self.p4_out_node = _dw_node()
        self.p5_out_node = _dw_node()
        self.p6_out_node = _dw_node()
        self.p7_out_node = _dw_node()

        # ---- 融合权重(原实现为普通 tensor,改为 Parameter 才能训练/保存/搬 GPU)----
        # 每节点一组可学习标量权重,relu 约束非负后做归一化加权(与 EfficientDet 一致)
        def _w(n):
            return nn.ParameterList(
                [nn.Parameter(torch.ones(1), requires_grad=True)
                 for _ in range(n)])

        self.p6_td_w = _w(2)
        self.p5_td_w = _w(2)
        self.p4_td_w = _w(2)
        self.p3_out_w = _w(2)
        self.p4_out_w = _w(3)
        self.p5_out_w = _w(3)
        self.p6_out_w = _w(3)
        self.p7_out_w = _w(2)

        # ---- 上采样 / 下采样 ----
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_downsample = nn.MaxPool2d(kernel_size=2)
        self.p4_downsample = nn.MaxPool2d(kernel_size=2)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2)
        self.p6_downsample = nn.MaxPool2d(kernel_size=2)

    def _fuse(self, weights, feats):
        """归一化加权融合: sum(w_i * f_i) / (sum(w_j) + eps),w 经 relu 保证非负.

        第一个 feats 是当前层的原始特征(目标尺寸),其余 resize 后的特征
        (上/下采样 2x 在奇偶尺寸下会有 1 像素偏差,如 800 输入 P6=25 ->
        maxpool=12 -> upsample=24)统一 pad 到目标尺寸再融合.
        """
        t_h, t_w = feats[0].shape[-2:]
        if isinstance(feats[0], torch.fx.Proxy):
            # fx 追踪 (QAT): 形状是 Proxy, 不能条件判断, 无条件 pad
            # (pad=0 时等价恒等, 运行期 pad 量按实际形状计算)
            feats = [
                F.pad(f, (0, t_w - f.shape[-1], 0, t_h - f.shape[-2]))
                for f in feats
            ]
        else:
            feats = [
                f if f.shape[-2:] == (t_h, t_w) else
                F.pad(f, (0, t_w - f.shape[-1], 0, t_h - f.shape[-2]))
                for f in feats
            ]
        w = [wi.relu() + 1e-4 for wi in weights]
        s = sum(wi for wi in w)
        return sum(wi * f for wi, f in zip(w, feats)) / s

    def forward(self, inputs):
        """inputs: [P3, P4, P5, P6] (backbone 4 层),返回 [P3..P7] 共 5 层."""
        # 关键:整个 neck 强制 fp32。DDP 下 SyncBatchNorm 走 sync_batch_norm
        # 路径,不在 autocast 的 fp32 排除列表内,会以 fp16 计算 batch 统计量;
        # 特征值较大时 sumsq 溢出 fp16(>65504) -> var=inf -> 输出全 NaN。
        # 外层 AmpOptimWrapper 的 autocast 对这里不生效,neck 用 fp32 计算,
        # 输出给下游 RPN/head 时再按各自 op 正常走混合精度。
        if next(self.parameters()).dtype == torch.float16:
            # PTQ fp16 推理: 权重已转 half, 不再强制 fp32 (训练时权重为
            # fp32, 仍走 autocast(enabled=False) 保护 SyncBN 统计量精度)
            return self._forward_impl(inputs)
        with autocast(enabled=False):
            return self._forward_impl(inputs)

    def _forward_impl(self, inputs):
        P3, P4, P5, P6 = inputs

        # P7 由 P6 下采样生成(stride 32 -> 64)
        P7 = self.p6_downsample(P6)

        # ---- top-down path ----
        P7_td = self.p7_td_conv(P7)
        P6_td_inp = self.p6_td_conv(P6)
        P6_td = self.p6_td_node(self._fuse(
            self.p6_td_w,
            [P6_td_inp, self.p2_upsample(P7_td)]))

        P5_td_inp = self.p5_td_conv(P5)
        P5_td = self.p5_td_node(self._fuse(
            self.p5_td_w,
            [P5_td_inp, self.p3_upsample(P6_td)]))

        P4_td_inp = self.p4_td_conv(P4)
        P4_td = self.p4_td_node(self._fuse(
            self.p4_td_w,
            [P4_td_inp, self.p4_upsample(P5_td)]))

        P3_td = self.p3_out_conv(P3)
        P3_out = self.p3_out_node(self._fuse(
            self.p3_out_w,
            [P3_td, self.p5_upsample(P4_td)]))

        # ---- bottom-up path ----
        P4_out = self.p4_out_node(self._fuse(
            self.p4_out_w,
            [P4_td_inp, P4_td, self.p3_downsample(P3_out)]))

        P5_out = self.p5_out_node(self._fuse(
            self.p5_out_w,
            [P5_td_inp, P5_td, self.p4_downsample(P4_out)]))

        P6_out = self.p6_out_node(self._fuse(
            self.p6_out_w,
            [P6_td_inp, P6_td, self.p5_downsample(P5_out)]))

        P7_out = self.p7_out_node(self._fuse(
            self.p7_out_w,
            [P7_td, self.p6_downsample(P6_out)]))

        return [P3_out, P4_out, P5_out, P6_out, P7_out]
