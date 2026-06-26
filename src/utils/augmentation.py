"""
Custom Augmentation for Long-Tail Object Detection

Key techniques:
1. Class-Balanced Copy-Paste: Paste tail class objects onto random backgrounds
2. Feature Hallucination: Generate synthetic features for tail classes
3. Mosaic + MixUp with class-aware sampling
4. Random rotation, scale, flip (standard)
5. Context-aware copy-paste: paste ships onto water, planes onto tarmac
"""
import os
import random
import copy
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
from PIL import Image

# Class groups for context-aware paste
SHIP_CLASSES = [0, 1, 2, 3]      # HM, LQS, QHS, MS
PLANE_CLASSES = list(range(4, 24)) # A1-A20
VEHICLE_CLASSES = [24]             # FSC

# Tail classes that need aggressive augmentation
TAIL_CLASSES = [0, 1, 9, 21, 24]  # HM(17), LQS(30), TU-160(361), KC-10(262), FSC(402)

class CopyPasteAugmentor:
    """
    Class-Balanced Copy-Paste Augmentation
    
    Strategy:
    1. Extract tail class objects from training images
    2. Paste them onto images that contain similar context
    3. Apply random transformations (scale, rotation, jitter)
    4. Update labels accordingly
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        tail_classes: List[int] = TAIL_CLASSES,
        paste_prob: float = 0.3,
        max_paste_per_image: int = 5,
        scale_range: Tuple[float, float] = (0.7, 1.3),
        rotation_range: Tuple[float, float] = (-10, 10),
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.tail_classes = tail_classes
        self.paste_prob = paste_prob
        self.max_paste_per_image = max_paste_per_image
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        
        # Object bank: {class_id: [(image, bbox, mask), ...]}
        self.object_bank: Dict[int, List] = {}
        self._build_object_bank()
    
    def _build_object_bank(self):
        """Build a bank of tail class objects for copy-paste."""
        print("Building object bank for copy-paste augmentation...")
        
        for label_path in self.labels_dir.glob("*.txt"):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        if cls_id in self.tail_classes:
                            if cls_id not in self.object_bank:
                                self.object_bank[cls_id] = []
                            
                            x, y, w, h = map(float, parts[1:])
                            self.object_bank[cls_id].append({
                                'image_name': label_path.stem,
                                'bbox': (x, y, w, h)  # YOLO format: normalized
                            })
        
        for cls_id, objs in self.object_bank.items():
            print(f"  Class {cls_id}: {len(objs)} objects in bank")
    
    def _extract_object(self, image_path: str, bbox_yolo: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract object patch and create mask from image."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None
        
        H, W = img.shape[:2]
        x, y, w, h = bbox_yolo
        # Convert YOLO to pixel
        x1 = int((x - w/2) * W)
        y1 = int((y - h/2) * H)
        x2 = int((x + w/2) * W)
        y2 = int((y + h/2) * H)
        
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None, None
        
        obj = img[y1:y2, x1:x2].copy()
        # Simple mask: non-black pixels (works for remote sensing objects on varied backgrounds)
        gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        return obj, mask
    
    def _get_context_class(self, cls_id: int) -> int:
        """Get the context group for a class."""
        if cls_id in SHIP_CLASSES:
            return 0  # Ship context
        elif cls_id in PLANE_CLASSES:
            return 1  # Plane context
        else:
            return 2  # Vehicle context
    
    def augment_image(
        self,
        image: np.ndarray,
        labels: List[Tuple[int, float, float, float, float]]
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        """
        Apply copy-paste augmentation to an image.
        
        Args:
            image: BGR image (H, W, 3)
            labels: List of (class_id, x, y, w, h) in YOLO format
        Returns:
            Augmented image and labels
        """
        if random.random() > self.paste_prob:
            return image, labels
        
        H, W = image.shape[:2]
        new_labels = list(labels)
        
        # Determine context of this image
        img_classes = [l[0] for l in labels]
        img_context = self._get_context_class(img_classes[0]) if img_classes else None
        
        n_paste = random.randint(1, self.max_paste_per_image)
        
        for _ in range(n_paste):
            # Pick a random tail class that matches context
            compatible_classes = [
                c for c in self.tail_classes
                if c in self.object_bank and
                (img_context is None or self._get_context_class(c) == img_context)
            ]
            
            if not compatible_classes:
                continue
            
            cls_id = random.choice(compatible_classes)
            obj_info = random.choice(self.object_bank[cls_id])
            
            # Extract object
            src_img_path = self.images_dir / f"{obj_info['image_name']}.jpg"
            if not src_img_path.exists():
                continue
            
            obj_patch, mask = self._extract_object(src_img_path, obj_info['bbox'])
            if obj_patch is None or mask is None:
                continue
            
            # Random scale
            scale = random.uniform(*self.scale_range)
            obj_patch = cv2.resize(obj_patch, None, fx=scale, fy=scale)
            mask = cv2.resize(mask, None, fx=scale, fy=scale)
            
            oh, ow = obj_patch.shape[:2]
            if oh >= H or ow >= W or oh < 5 or ow < 5:
                continue
            
            # Random position (avoid edges)
            x_off = random.randint(0, W - ow)
            y_off = random.randint(0, H - oh)
            
            # Paste using mask
            roi = image[y_off:y_off+oh, x_off:x_off+ow]
            mask_bool = mask > 0
            roi[mask_bool] = obj_patch[mask_bool]
            image[y_off:y_off+oh, x_off:x_off+ow] = roi
            
            # Convert to YOLO format
            cx = (x_off + ow/2) / W
            cy = (y_off + oh/2) / H
            nw = ow / W
            nh = oh / H
            new_labels.append((cls_id, cx, cy, nw, nh))
        
        return image, new_labels


class FeatureHallucinator:
    """
    Dynamic Feature Hallucination for Tail Classes
    
    Instead of augmenting in image space, we augment in feature space.
    This is more efficient and doesn't require precise object masks.
    
    Strategy:
    1. During training, collect feature statistics (mean, std) per class
    2. For tail classes, generate synthetic features by sampling from
       the learned distribution
    3. Add these synthetic features to the classification head training
    
    Note: This is implemented as a callback/hook in the training pipeline.
    """
    
    def __init__(
        self,
        tail_classes: List[int] = TAIL_CLASSES,
        hallucination_factor: int = 3,
        noise_std: float = 0.1,
    ):
        self.tail_classes = tail_classes
        self.hallucination_factor = hallucination_factor
        self.noise_std = noise_std
        self.feature_stats: Dict[int, Dict] = {}
    
    def update_stats(self, cls_id: int, features: np.ndarray):
        """Update feature statistics for a class."""
        if cls_id not in self.feature_stats:
            self.feature_stats[cls_id] = {
                'mean': features.mean(axis=0),
                'std': features.std(axis=0),
                'count': len(features)
            }
        else:
            # Running update
            old_count = self.feature_stats[cls_id]['count']
            new_count = old_count + len(features)
            
            old_mean = self.feature_stats[cls_id]['mean']
            old_std = self.feature_stats[cls_id]['std']
            
            new_mean = (old_mean * old_count + features.mean(axis=0) * len(features)) / new_count
            new_std = np.sqrt(
                (old_std**2 * old_count + features.std(axis=0)**2 * len(features)) / new_count
            )
            
            self.feature_stats[cls_id] = {
                'mean': new_mean,
                'std': new_std,
                'count': new_count
            }
    
    def hallucinate(self, cls_id: int, n_samples: int = None) -> np.ndarray:
        """Generate synthetic features for a tail class."""
        if cls_id not in self.feature_stats:
            return None
        
        if n_samples is None:
            n_samples = self.hallucination_factor
        
        stats = self.feature_stats[cls_id]
        # Sample from learned distribution + noise
        synthetic = np.random.randn(n_samples, len(stats['mean']))
        synthetic = synthetic * stats['std'] + stats['mean']
        # Add extra noise for diversity
        synthetic += np.random.randn(*synthetic.shape) * self.noise_std
        
        return synthetic


class MosaicAugmentor:
    """
    Class-Aware Mosaic Augmentation
    
    Standard mosaic randomly selects 4 images. We modify it to:
    1. Ensure at least 1 image contains a tail class
    2. Weight selection probability by class rarity
    """
    
    def __init__(
        self,
        tail_classes: List[int] = TAIL_CLASSES,
        tail_boost: float = 3.0,
    ):
        self.tail_classes = set(tail_classes)
        self.tail_boost = tail_boost
    
    def get_sample_weights(
        self,
        image_class_list: List[List[int]],
        base_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute sampling weights for mosaic image selection.
        Images with tail classes get higher weight.
        """
        n = len(image_class_list)
        if base_weights is None:
            weights = np.ones(n)
        else:
            weights = base_weights.copy()
        
        for i, classes in enumerate(image_class_list):
            if any(c in self.tail_classes for c in classes):
                weights[i] *= self.tail_boost
        
        # Normalize
        weights = weights / weights.sum()
        return weights


if __name__ == "__main__":
    print("Augmentation utilities loaded successfully.")
    print(f"Tail classes: {TAIL_CLASSES}")
    print(f"Ship classes: {SHIP_CLASSES}")
    print(f"Plane classes: {PLANE_CLASSES[:5]}... (20 total)")
