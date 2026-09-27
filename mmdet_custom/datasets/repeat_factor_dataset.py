"""
Repeat Factor Sampling (RFS) dataset wrapper for long-tailed detection.

Oversamples images containing rare categories based on category frequency.
From LVIS / Detectron2 / Balanced Group Softmax (CVPR 2020).
"""

import math
import numpy as np
from mmdet.registry import DATASETS
from torch.utils.data import Dataset


@DATASETS.register_module()
class RepeatFactorDataset(Dataset):
    """Dataset wrapper that repeats images with rare categories.

    For each category c, compute f(c) = fraction of images containing c.
    Repeat factor r(c) = max(1, sqrt(t / f(c))).
    Image repeat factor = max(r(c) for c in image).

    Args:
        dataset (dict): Config for the base dataset.
        repeat_thr (float): Frequency threshold t. Default 0.001.
        resample_classes (list, optional): Only apply repeat factors to these
            class ids; other classes get factor 1. None = all classes.
    """

    def __init__(self, dataset, repeat_thr=0.001, resample_classes=None):
        self.dataset = DATASETS.build(dataset)
        self.repeat_thr = repeat_thr
        self.resample_classes = resample_classes
        self.CLASSES = self.dataset.metainfo.get('classes', [])
        self.metainfo = self.dataset.metainfo
        self.cat2label = getattr(self.dataset, 'cat2label', {})

        # Compute per-category frequency
        category_image_count = self._compute_category_frequency()
        self.repeat_factors = self._compute_repeat_factors(category_image_count)
        self.repeat_indices = self._build_repeat_indices()

    def _compute_category_frequency(self):
        """Count images containing each category."""
        num_images = len(self.dataset)
        num_cats = len(self.CLASSES)
        cat_image_count = np.zeros(num_cats, dtype=np.float64)

        for idx in range(num_images):
            data_info = self.dataset.get_data_info(idx)
            if 'instances' in data_info:
                labels = set()
                for inst in data_info['instances']:
                    # Skip ignore/iscrowd annotations
                    if inst.get('ignore', False) or inst.get('iscrowd', 0):
                        continue
                    if 'bbox_label' in inst:
                        labels.add(inst['bbox_label'])
                for c in labels:
                    if 0 <= c < num_cats:
                        cat_image_count[c] += 1

        return cat_image_count

    def _compute_repeat_factors(self, cat_image_count):
        """Compute per-category repeat factors, then per-image."""
        num_images = len(self.dataset)
        num_cats = len(self.CLASSES)

        # Category frequency
        freq = cat_image_count / max(num_images, 1)
        # Repeat factor per category: max(1, sqrt(t / f))
        cat_repeat = np.maximum(1.0, np.sqrt(self.repeat_thr / freq.clip(min=1e-10)))
        cat_repeat = np.minimum(cat_repeat, 10.0)  # cap at 10x

        # Per-image repeat factor = max of its categories (skip ignore)
        # 若指定 resample_classes, 只在这些类别上取 max, 其余类别视为 1
        img_repeat = np.ones(num_images, dtype=np.float64)
        for idx in range(num_images):
            data_info = self.dataset.get_data_info(idx)
            if 'instances' in data_info:
                max_r = 1.0
                for inst in data_info['instances']:
                    if inst.get('ignore', False) or inst.get('iscrowd', 0):
                        continue
                    if 'bbox_label' in inst:
                        c = inst['bbox_label']
                        if 0 <= c < num_cats:
                            if self.resample_classes is None or c in self.resample_classes:
                                max_r = max(max_r, cat_repeat[c])
                img_repeat[idx] = max_r

        return img_repeat

    def _build_repeat_indices(self):
        """Build expanded index list with repeats."""
        indices = []
        for idx in range(len(self.dataset)):
            r = int(math.ceil(self.repeat_factors[idx]))
            indices.extend([idx] * r)
        return indices

    def __len__(self):
        return len(self.repeat_indices)

    def __getitem__(self, idx):
        orig_idx = self.repeat_indices[idx]
        return self.dataset[orig_idx]

    def get_data_info(self, idx):
        orig_idx = self.repeat_indices[idx]
        return self.dataset.get_data_info(orig_idx)

    def full_init(self):
        self.dataset.full_init()
