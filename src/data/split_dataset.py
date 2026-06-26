"""
Dataset Splitting Script
Splits train data into train/val by source image ID to prevent data leakage.
Also creates the proper directory structure for YOLO training.
"""
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = Path(r"C:\Users\lenovo\Desktop\数据集\data")
IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
LABELS_TRAIN = DATASET_ROOT / "labels" / "train"

# Output split directories
IMAGES_VAL = DATASET_ROOT / "images" / "val"
LABELS_VAL = DATASET_ROOT / "labels" / "val"

SEED = 42
TRAIN_RATIO = 0.8

def get_source_image_id(filename):
    """Extract source image ID from crop filename."""
    name = Path(filename).stem
    if "_crop" in name:
        name = name.rsplit("_crop", 1)[0]
    return name

def split_dataset():
    """Split dataset by source image ID."""
    random.seed(SEED)
    
    # Ensure val directories exist
    IMAGES_VAL.mkdir(parents=True, exist_ok=True)
    LABELS_VAL.mkdir(parents=True, exist_ok=True)
    
    # Group by source image
    source_groups = defaultdict(list)
    for img_path in IMAGES_TRAIN.glob("*.jpg"):
        src_id = get_source_image_id(img_path.name)
        source_groups[src_id].append(img_path.stem)
    
    source_ids = list(source_groups.keys())
    random.shuffle(source_ids)
    
    n_train = int(len(source_ids) * TRAIN_RATIO)
    val_sources = set(source_ids[n_train:])
    
    print(f"Total source images: {len(source_ids)}")
    print(f"Train sources: {n_train}")
    print(f"Val sources: {len(val_sources)}")
    
    # Move val files
    moved_count = 0
    for src_id in source_ids:
        if src_id in val_sources:
            for stem in source_groups[src_id]:
                # Move image
                img_src = IMAGES_TRAIN / f"{stem}.jpg"
                img_dst = IMAGES_VAL / f"{stem}.jpg"
                if img_src.exists():
                    shutil.move(str(img_src), str(img_dst))
                
                # Move label
                lbl_src = LABELS_TRAIN / f"{stem}.txt"
                lbl_dst = LABELS_VAL / f"{stem}.txt"
                if lbl_src.exists():
                    shutil.move(str(lbl_src), str(lbl_dst))
                
                moved_count += 1
    
    print(f"Moved {moved_count} image-label pairs to val/")
    
    # Count remaining
    train_imgs = len(list(IMAGES_TRAIN.glob("*.jpg")))
    val_imgs = len(list(IMAGES_VAL.glob("*.jpg")))
    print(f"\nFinal split:")
    print(f"  Train: {train_imgs} images")
    print(f"  Val: {val_imgs} images")

if __name__ == "__main__":
    split_dataset()
