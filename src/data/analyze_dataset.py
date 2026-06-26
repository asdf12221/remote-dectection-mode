"""
Dataset Analysis Script
Analyzes class distribution, image sizes, and provides splitting strategy.
"""
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
import random

import numpy as np
from PIL import Image

DATASET_ROOT = Path(r"C:\Users\lenovo\Desktop\数据集\data")
IMAGES_DIR = DATASET_ROOT / "images" / "train"
LABELS_DIR = DATASET_ROOT / "labels" / "train"

CLASS_NAMES = {
    0: "HM(航母)", 1: "LQS(两栖舰)", 2: "QHS(驱护舰)", 3: "MS(民船)",
    4: "SU-35", 5: "C-130", 6: "C-17", 7: "C-5", 8: "F-16", 9: "TU-160",
    10: "E-3", 11: "B-52", 12: "P-3C", 13: "B-1B", 14: "E-8",
    15: "TU-22", 16: "F-15", 17: "KC-135", 18: "F-22", 19: "FA-18",
    20: "TU-95", 21: "KC-10", 22: "SU-34", 23: "SU-24", 24: "FSC(发射车)"
}

def parse_label(label_path):
    """Parse YOLO format label file."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                boxes.append((cls_id, x, y, w, h))
    return boxes

def analyze_distribution():
    """Analyze class distribution in the dataset."""
    class_counts = Counter()
    image_classes = defaultdict(set)
    image_boxes_count = defaultdict(int)
    image_sizes = []
    
    label_files = list(LABELS_DIR.glob("*.txt"))
    print(f"Total label files: {len(label_files)}")
    
    for i, label_path in enumerate(label_files):
        boxes = parse_label(label_path)
        img_name = label_path.stem
        for cls_id, *_ in boxes:
            class_counts[cls_id] += 1
            image_classes[img_name].add(cls_id)
        image_boxes_count[img_name] = len(boxes)
    
    # Check image sizes (sample)
    image_files = list(IMAGES_DIR.glob("*.jpg"))
    sample_size = min(200, len(image_files))
    for img_path in random.sample(image_files, sample_size):
        try:
            with Image.open(img_path) as img:
                image_sizes.append(img.size)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
    
    # Print distribution
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION")
    print("="*70)
    total = sum(class_counts.values())
    for cls_id in range(25):
        count = class_counts[cls_id]
        name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
        pct = count / total * 100 if total > 0 else 0
        bar = "#" * int(pct * 2)
        print(f"  {cls_id:2d} {name:20s} | {count:5d} ({pct:5.2f}%) {bar}")
    print(f"\n  Total boxes: {total}")
    
    # Imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    print(f"\n  Imbalance ratio (max/min): {max_count}:{min_count} = {max_count/min_count:.1f}:1")
    
    # Class groups
    head = [c for c in range(25) if class_counts[c] >= 1000]
    medium = [c for c in range(25) if 200 <= class_counts[c] < 1000]
    tail = [c for c in range(25) if class_counts[c] < 200]
    print(f"\n  Head classes (>=1000): {len(head)} classes")
    print(f"  Medium classes (200-999): {len(medium)} classes")
    print(f"  Tail classes (<200): {len(tail)} classes")
    print(f"  Tail class IDs: {tail}")
    
    # Image sizes
    if image_sizes:
        sizes_counter = Counter(image_sizes)
        print("\n" + "="*70)
        print("IMAGE SIZE DISTRIBUTION (sampled)")
        print("="*70)
        for size, count in sizes_counter.most_common(10):
            print(f"  {size[0]}x{size[1]}: {count} images")
    
    # Boxes per image
    box_counts = list(image_boxes_count.values())
    print("\n" + "="*70)
    print("BOXES PER IMAGE")
    print("="*70)
    print(f"  Mean: {np.mean(box_counts):.2f}")
    print(f"  Median: {np.median(box_counts):.2f}")
    print(f"  Min: {min(box_counts)}, Max: {max(box_counts)}")
    
    # Multi-class images
    multi_class = sum(1 for cls_set in image_classes.values() if len(cls_set) >= 2)
    print(f"\n  Multi-class images: {multi_class} ({multi_class/len(image_classes)*100:.1f}%)")
    
    return class_counts

def get_source_image_id(filename):
    """Extract source image ID from crop filename for grouped splitting.
    
    Example: 01-PAN-20240418-318-232-L00000010061-CCD14_3_crop1.jpg
    -> Source ID: 01-PAN-20240418-318-232-L00000010061-CCD14_3
    """
    # Remove extension
    name = Path(filename).stem
    # Remove _cropN suffix
    if "_crop" in name:
        name = name.rsplit("_crop", 1)[0]
    return name

def plan_split(train_ratio=0.8, seed=42):
    """Plan train/val split grouped by source image to prevent leakage."""
    random.seed(seed)
    
    label_files = list(LABELS_DIR.glob("*.txt"))
    
    # Group by source image
    source_groups = defaultdict(list)
    for lf in label_files:
        src_id = get_source_image_id(lf.name)
        source_groups[src_id].append(lf.stem)
    
    source_ids = list(source_groups.keys())
    random.shuffle(source_ids)
    
    n_train = int(len(source_ids) * train_ratio)
    train_sources = set(source_ids[:n_train])
    val_sources = set(source_ids[n_train:])
    
    train_files = []
    val_files = []
    for src_id in source_ids:
        if src_id in train_sources:
            train_files.extend(source_groups[src_id])
        else:
            val_files.extend(source_groups[src_id])
    
    print(f"\n" + "="*70)
    print(f"SPLIT PLAN (seed={seed}, train_ratio={train_ratio})")
    print("="*70)
    print(f"  Source images: {len(source_ids)}")
    print(f"  Train: {len(train_files)} images from {len(train_sources)} sources")
    print(f"  Val: {len(val_files)} images from {len(val_sources)} sources")
    
    # Check class distribution in each split
    train_class_counts = Counter()
    val_class_counts = Counter()
    for fname in train_files:
        boxes = parse_label(LABELS_DIR / f"{fname}.txt")
        for cls_id, *_ in boxes:
            train_class_counts[cls_id] += 1
    for fname in val_files:
        boxes = parse_label(LABELS_DIR / f"{fname}.txt")
        for cls_id, *_ in boxes:
            val_class_counts[cls_id] += 1
    
    print(f"\n  Class distribution comparison:")
    print(f"  {'Class':<20s} {'Train':>8s} {'Val':>8s} {'Ratio':>8s}")
    for cls_id in range(25):
        name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
        tr = train_class_counts[cls_id]
        va = val_class_counts[cls_id]
        ratio = tr / (tr + va) * 100 if (tr + va) > 0 else 0
        print(f"  {name:<20s} {tr:8d} {va:8d} {ratio:7.1f}%")
    
    return train_files, val_files

if __name__ == "__main__":
    print("="*70)
    print("XH-202625 Dataset Analysis")
    print("="*70)
    
    class_counts = analyze_distribution()
    train_files, val_files = plan_split(train_ratio=0.8, seed=42)
    
    print("\nAnalysis complete!")
