"""
Create oversampled dataset for tail classes.
Generates a new dataset where images containing tail classes are duplicated
to balance the class distribution.
"""
import os
import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict

DATASET_ROOT = Path(r"C:\Users\lenovo\Desktop\数据集\data")
IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
LABELS_TRAIN = DATASET_ROOT / "labels" / "train"

# Tail classes (less than 200 boxes)
TAIL_CLASSES = [0, 1, 9, 21, 24]  # HM, LQS, TU-160, KC-10, FSC

# Oversample factors based on inverse frequency
OVERSAMPLE_FACTORS = {
    0: 30,   # HM: 17 boxes -> x30 = 510 images
    1: 20,   # LQS: 30 boxes -> x20 = 600 images  
    9: 3,    # TU-160: 361 boxes -> x3
    21: 4,   # KC-10: 262 boxes -> x4
    24: 3,   # FSC: 402 boxes -> x3
}

def create_oversampled_dataset():
    """Create oversampled dataset for tail classes."""
    output_images = DATASET_ROOT / "images" / "train_balanced"
    output_labels = DATASET_ROOT / "labels" / "train_balanced"
    
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # First: copy all original files
    print("Copying original files...")
    for img_path in IMAGES_TRAIN.glob("*.jpg"):
        shutil.copy2(str(img_path), str(output_images / img_path.name))
    
    for lbl_path in LABELS_TRAIN.glob("*.txt"):
        shutil.copy2(str(lbl_path), str(output_labels / lbl_path.name))
    
    # Then: duplicate tail class images
    print("Oversampling tail class images...")
    
    # Find images containing tail classes
    tail_images = defaultdict(list)  # {cls_id: [image_names]}
    for lbl_path in LABELS_TRAIN.glob("*.txt"):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    if cls_id in OVERSAMPLE_FACTORS:
                        tail_images[cls_id].append(lbl_path.stem)
    
    total_copies = 0
    for cls_id, images in tail_images.items():
        factor = OVERSAMPLE_FACTORS[cls_id]
        unique_images = list(set(images))
        print(f"  Class {cls_id}: {len(unique_images)} unique images, oversample x{factor}")
        
        for i in range(1, factor):  # i=0 is the original
            for img_name in unique_images:
                src_img = IMAGES_TRAIN / f"{img_name}.jpg"
                src_lbl = LABELS_TRAIN / f"{img_name}.txt"
                
                if src_img.exists() and src_lbl.exists():
                    dst_img = output_images / f"{img_name}_os{i}.jpg"
                    dst_lbl = output_labels / f"{img_name}_os{i}.txt"
                    shutil.copy2(str(src_img), str(dst_img))
                    shutil.copy2(str(src_lbl), str(dst_lbl))
                    total_copies += 1
    
    # Count final
    n_images = len(list(output_images.glob("*.jpg")))
    n_labels = len(list(output_labels.glob("*.txt")))
    
    print(f"\nOversampling complete!")
    print(f"  Original: {len(list(IMAGES_TRAIN.glob('*.jpg')))} images")
    print(f"  Balanced: {n_images} images ({total_copies} copies added)")
    print(f"  Labels: {n_labels}")
    
    # Update dataset.yaml
    import yaml
    yaml_path = DATASET_ROOT / "dataset_balanced.yaml"
    config = {
        'path': str(DATASET_ROOT).replace('\\', '/'),
        'train': 'images/train_balanced',
        'val': 'images/val',
        'nc': 25,
        'names': {
            0: 'HM', 1: 'LQS', 2: 'QHS', 3: 'MS',
            4: 'A1_SU-35', 5: 'A2_C-130', 6: 'A3_C-17', 7: 'A4_C-5',
            8: 'A5_F-16', 9: 'A6_TU-160', 10: 'A7_E-3', 11: 'A8_B-52',
            12: 'A9_P-3C', 13: 'A10_B-1B', 14: 'A11_E-8',
            15: 'A12_TU-22', 16: 'A13_F-15', 17: 'A14_KC-135',
            18: 'A15_F-22', 19: 'A16_FA-18', 20: 'A17_TU-95',
            21: 'A18_KC-10', 22: 'A19_SU-34', 23: 'A20_SU-24', 24: 'FSC'
        }
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"  Config: {yaml_path}")

if __name__ == "__main__":
    create_oversampled_dataset()
