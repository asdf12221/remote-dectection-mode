# CRTDetector: cRT (Classifier Re-Training, Kang et al. ICLR'20) 专用检测器
# - 在 __init__ 中冻结除分类器 (fc_cls) 外的所有参数:
#   backbone / FPN / RPN / bbox_head 共享卷积与回归分支 全部冻结
# - 必须早于 optimizer 构建: CustomLayerDecayOptimizerConstructor
#   会跳过 requires_grad=False 的参数 → 冻结后不进 optimizer, 不占 lr
# - load_from 只覆盖权重值, 不改变 requires_grad → 冻结不受加载影响
from mmdet.models.detectors import CascadeRCNN
from mmdet.registry import MODELS
from mmengine.logging import MMLogger


@MODELS.register_module()
class CRTDetector(CascadeRCNN):
    """Cascade R-CNN for Classifier Re-Training (cRT).

    除 bbox_head 的 fc_cls 外全部冻结, 配合类别均衡重采样
    (RepeatFactorDataset) 只重训分类器, 修复尾部类别的分类边界。
    """

    def __init__(self, unfreeze_substr=('fc_cls', ), **kwargs):
        super().__init__(**kwargs)
        logger = MMLogger.get_current_instance()
        kept, frozen = [], []
        for name, param in self.named_parameters():
            if any(k in name for k in unfreeze_substr):
                param.requires_grad_(True)
                kept.append(name)
            else:
                param.requires_grad_(False)
                frozen.append(name)
        logger.info('[cRT] unfrozen (%d): %s', len(kept), ', '.join(kept))
        logger.info('[cRT] frozen (%d), e.g.: %s ...', len(frozen),
                    ', '.join(frozen[:5]))
