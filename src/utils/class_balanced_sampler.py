"""
Class-Balanced Sampling for Long-Tail Object Detection
Implements: inverse frequency sampling, square-root sampling, class-balanced repeat factor sampling
"""
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional

class ClassBalancedSampler:
    """
    Class-balanced sampler for object detection.
    
    Strategies:
    1. inverse_freq: weight_i = 1 / freq_i
    2. sqrt_inv_freq: weight_i = 1 / sqrt(freq_i)  
    3. balanced_repeat: repeat_factor_i = max(1, floor(median_freq / freq_i))
    4. progressivereweight: gradually increase tail class weights during training
    """
    
    def __init__(
        self,
        class_counts: Dict[int, int],
        strategy: str = "sqrt_inv_freq",
        beta: float = 0.9999,
        effective_num: float = 1.0
    ):
        self.class_counts = class_counts
        self.n_classes = len(class_counts)
        self.strategy = strategy
        self.beta = beta
        self.effective_num = effective_num
        
        # Compute class weights
        self.weights = self._compute_weights()
        
        # Per-sample weights for each image
        self.sample_weights = None
        
    def _compute_weights(self) -> np.ndarray:
        """Compute per-class weights based on strategy."""
        counts = np.array([self.class_counts.get(i, 1) for i in range(self.n_classes)])
        
        if self.strategy == "inverse_freq":
            # Basic inverse frequency
            weights = 1.0 / counts
            
        elif self.strategy == "sqrt_inv_freq":
            # Square root inverse (less aggressive)
            weights = 1.0 / np.sqrt(counts)
            
        elif self.strategy == "balanced_repeat":
            # Class-balanced repeat factor (per image, not per box)
            median_freq = np.median(counts)
            weights = median_freq / counts
            weights = np.clip(weights, 1.0, 10.0)  # Cap max repeat at 10x
            
        elif self.strategy == "class_balanced":
            # Class-balanced loss weights (CUHK/Segal et al.)
            effective_num = 1.0 - np.power(self.beta, counts)
            weights = (1.0 - self.beta) / effective_num
            
        elif self.strategy == "progressivereweight":
            # Progressively increase tail class importance
            # Weights will be modulated by epoch during training
            weights = 1.0 / np.sqrt(counts)
            
        else:
            weights = np.ones(self.n_classes)
        
        # Normalize
        weights = weights / weights.sum() * self.n_classes
        
        return weights
    
    def get_sample_weights(self, image_class_ids: List[int]) -> np.ndarray:
        """
        Compute per-sample weights for a batch of images.
        
        Args:
            image_class_ids: List of class IDs present in each image
        Returns:
            Array of weights for each image (shape: n_images,)
        """
        # For each image, use the max weight among its classes
        img_weights = []
        for cls_list in image_class_ids:
            if cls_list:
                max_w = max(self.weights[c] for c in cls_list if c < self.n_classes)
            else:
                max_w = 1.0
            img_weights.append(max_w)
        return np.array(img_weights)
    
    def get_class_weights(self) -> np.ndarray:
        """Return the computed class weights."""
        return self.weights.copy()
    
    def get_repeat_factors(self, class_counts_per_image: List[Dict[int, int]]) -> List[float]:
        """
        Compute repeat factor for each image based on its class content.
        Used in Repeat Factor Sampling (Shen et al.).
        
        Args:
            class_counts_per_image: List of {class_id: box_count} dicts for each image
        Returns:
            List of repeat factors (float) for each image
        """
        # First compute per-image class repeat factors
        img_repeat_factors = []
        for img_class_dict in class_counts_per_image:
            factors = []
            for cls_id, box_count in img_class_dict.items():
                rf = np.sqrt(self.weights[cls_id]) if cls_id < self.n_classes else 1.0
                # Use max per class in image
                factors.append(rf)
            img_repeat_factors.append(max(factors) if factors else 1.0)
        
        return img_repeat_factors


class BalancedGroupSoftmax:
    """
    Balanced Group Softmax (BAGS) for long-tail detection.
    From: Li et al., CVPR 2020 Oral.
    
    Groups classes by sample count, then performs softmax within each group.
    This prevents head classes from dominating the softmax.
    """
    
    def __init__(self, class_counts: Dict[int, int], n_groups: int = 3):
        self.class_counts = class_counts
        self.n_groups = n_groups
        self.group_assignment = self._assign_groups()
        self.group_thresholds = self._compute_group_thresholds()
        
    def _assign_groups(self) -> Dict[int, int]:
        """Assign each class to a group based on sample count."""
        sorted_classes = sorted(
            self.class_counts.keys(),
            key=lambda c: self.class_counts[c],
            reverse=True
        )
        
        n_classes = len(sorted_classes)
        group_size = n_classes // self.n_groups
        
        assignment = {}
        for i, cls_id in enumerate(sorted_classes):
            if i < group_size:
                assignment[cls_id] = 0  # Head group
            elif i < 2 * group_size:
                assignment[cls_id] = 1  # Medium group
            else:
                assignment[cls_id] = 2  # Tail group
                
        # Ensure tail classes (very few samples) go to tail group
        for cls_id, count in self.class_counts.items():
            if count < 200:
                assignment[cls_id] = 2  # Force to tail
                
        return assignment
    
    def _compute_group_thresholds(self) -> Dict[int, float]:
        """Compute logit thresholds per group."""
        thresholds = {}
        for grp in range(self.n_groups):
            grp_classes = [c for c, g in self.group_assignment.items() if g == grp]
            if grp_classes:
                grp_counts = [self.class_counts[c] for c in grp_classes]
                # Threshold proportional to inverse of group size
                thresholds[grp] = np.log(np.mean(grp_counts))
        return thresholds
    
    def get_group(self, cls_id: int) -> int:
        """Get group ID for a class."""
        return self.group_assignment.get(cls_id, 2)
    
    def print_groups(self):
        """Print class group assignments."""
        for grp in range(self.n_groups):
            grp_classes = [c for c, g in self.group_assignment.items() if g == grp]
            grp_classes.sort(key=lambda c: self.class_counts[c], reverse=True)
            print(f"\nGroup {grp}:")
            for c in grp_classes:
                print(f"  Class {c}: {self.class_counts[c]} boxes")


def compute_focal_weights(class_counts: Dict[int, int], gamma: float = 2.0) -> np.ndarray:
    """
    Compute focal loss weights for class imbalance.
    
    FL(p) = -alpha * (1-p)^gamma * log(p)
    alpha_i = 1 / (n_classes * freq_i)
    """
    n_classes = len(class_counts)
    total = sum(class_counts.values())
    
    alphas = []
    for i in range(n_classes):
        count = class_counts.get(i, 1)
        freq = count / total
        # alpha balanced: total alpha per class normalized
        alpha_i = (1.0 / freq) / sum(1.0 / class_counts.get(j, 1) for j in range(n_classes))
        alphas.append(alpha_i)
    
    alphas = np.array(alphas)
    # Normalize so sum(alphas) = n_classes
    alphas = alphas / alphas.sum() * n_classes
    
    return alphas


if __name__ == "__main__":
    # Test with sample data
    sample_counts = {i: max(17, int(2147 * np.exp(-0.15 * i))) for i in range(25)}
    
    sampler = ClassBalancedSampler(sample_counts, strategy="sqrt_inv_freq")
    print("Class weights (sqrt_inv_freq):")
    for i in range(25):
        print(f"  Class {i}: {sampler.weights[i]:.3f}")
    
    bags = BalancedGroupSoftmax(sample_counts, n_groups=3)
    bags.print_groups()
    
    focal_w = compute_focal_weights(sample_counts)
    print("\nFocal loss alpha weights:")
    for i in range(25):
        print(f"  Class {i}: {focal_w[i]:.3f}")
