"""
离线 Copy-Paste 增强：将尾类目标裁剪后粘贴到其他图片上
不依赖mosaic，不增加显存，离线预处理
"""
import os
import cv2
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image

# 配置
TRAIN_IMG_DIR = r"C:\Users\lenovo\Desktop\数据集\data\images\train"
TRAIN_LBL_DIR = r"C:\Users\lenovo\Desktop\数据集\data\labels\train"
OUTPUT_IMG_DIR = r"C:\Users\lenovo\Desktop\数据集\data\images\train_cp"
OUTPUT_LBL_DIR = r"C:\Users\lenovo\Desktop\数据集\data\labels\train_cp"

# 尾类定义: class_id -> 需要粘贴的目标数量
TAIL_CLASSES = {
    0: 30,   # HM - 航母
    1: 25,   # LQS - 两栖舰
    21: 15,  # KC-10
    14: 10,  # E-8
    7: 8,    # C-5
    24: 8,   # FSC
}

def load_labels(lbl_path):
    """加载YOLO格式标签"""
    if not os.path.exists(lbl_path):
        return []
    labels = []
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                labels.append({'cls': cls, 'cx': cx, 'cy': cy, 'w': w, 'h': h})
    return labels

def extract_objects(img_dir, lbl_dir, target_classes):
    """从训练集中提取尾类目标区域"""
    objects = defaultdict(list)
    
    for img_file in sorted(Path(img_dir).glob('*.jpg')):
        lbl_file = Path(lbl_dir) / (img_file.stem + '.txt')
        labels = load_labels(str(lbl_file))
        if not labels:
            continue
        
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        for lbl in labels:
            if lbl['cls'] in target_classes:
                # 计算像素坐标
                x1 = int((lbl['cx'] - lbl['w']/2) * w)
                y1 = int((lbl['cy'] - lbl['h']/2) * h)
                x2 = int((lbl['cx'] + lbl['w']/2) * w)
                y2 = int((lbl['cy'] + lbl['h']/2) * h)
                
                # 扩展边界（带一些背景）
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                crop = img[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    objects[lbl['cls']].append({
                        'crop': crop,
                        'orig_w': x2 - x1,
                        'orig_h': y2 - y1,
                    })
    
    for cls_id, objs in objects.items():
        print(f"  Class {cls_id}: extracted {len(objs)} objects")
    return objects

def paste_object(base_img, obj_crop, base_labels, target_cls):
    """将一个目标粘贴到base_img上，返回新标签"""
    h, w = base_img.shape[:2]
    obj_h, obj_w = obj_crop.shape[:2]
    
    # 随机缩放 (0.7-1.3)
    scale = random.uniform(0.7, 1.3)
    new_w = int(obj_w * scale)
    new_h = int(obj_h * scale)
    if new_w >= w or new_h >= h or new_w < 10 or new_h < 10:
        return None, None
    
    resized = cv2.resize(obj_crop, (new_w, new_h))
    
    # 随机位置（确保在图像内）
    max_x = w - new_w
    max_y = h - new_h
    if max_x <= 0 or max_y <= 0:
        return None, None
    
    px = random.randint(0, max_x)
    py = random.randint(0, max_y)
    
    # 简单粘贴（无mask blending，直接覆盖）
    # 创建mask（非黑色区域为目标）
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # 提取目标区域
    roi = base_img[py:py+new_h, px:px+new_w]
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(resized, resized, mask=mask)
    blended = cv2.add(bg, fg)
    base_img[py:py+new_h, px:px+new_w] = blended
    
    # YOLO格式标签
    cx = (px + new_w / 2) / w
    cy = (py + new_h / 2) / h
    nw = new_w / w
    nh = new_h / h
    
    new_label = {'cls': target_cls, 'cx': cx, 'cy': cy, 'w': nw, 'h': nh}
    return base_img, new_label

def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)
    
    print("=" * 60)
    print("离线 Copy-Paste 增强")
    print("=" * 60)
    
    # Step 1: 提取尾类目标
    print("\nStep 1: 提取尾类目标...")
    objects = extract_objects(TRAIN_IMG_DIR, TRAIN_LBL_DIR, set(TAIL_CLASSES.keys()))
    
    total_extracted = sum(len(v) for v in objects.values())
    if total_extracted == 0:
        print("未提取到任何尾类目标，退出")
        return
    
    # Step 2: 对每张训练图，随机粘贴尾类目标
    print("\nStep 2: 粘贴增强...")
    img_files = sorted(Path(TRAIN_IMG_DIR).glob('*.jpg'))
    augmented_count = 0
    
    for img_file in img_files:
        lbl_file = Path(TRAIN_LBL_DIR) / (img_file.stem + '.txt')
        labels = load_labels(str(lbl_file))
        
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        new_labels = list(labels)  # 复制原有标签
        
        # 对每个尾类，按概率粘贴
        for cls_id, num_to_paste in TAIL_CLASSES.items():
            if cls_id not in objects or not objects[cls_id]:
                continue
            
            # 每张图有50%概率粘贴一个该类目标
            for _ in range(num_to_paste):
                if random.random() < 0.5:
                    obj = random.choice(objects[cls_id])
                    result_img, new_label = paste_object(img, obj['crop'], new_labels, cls_id)
                    if result_img is not None and new_label is not None:
                        img = result_img
                        new_labels.append(new_label)
        
        # 保存增强后的图片和标签
        out_img_path = Path(OUTPUT_IMG_DIR) / img_file.name
        out_lbl_path = Path(OUTPUT_LBL_DIR) / (img_file.stem + '.txt')
        
        cv2.imwrite(str(out_img_path), img)
        with open(out_lbl_path, 'w') as f:
            for lbl in new_labels:
                f.write(f"{lbl['cls']} {lbl['cx']:.6f} {lbl['cy']:.6f} {lbl['w']:.6f} {lbl['h']:.6f}\n")
        
        augmented_count += 1
    
    print(f"\n增强完成: {augmented_count} 张图片")
    print(f"输出目录: {OUTPUT_IMG_DIR}")
    
    # 统计增强后的类别分布
    print("\n增强后类别分布:")
    class_counts = defaultdict(int)
    for lbl_file in Path(OUTPUT_LBL_DIR).glob('*.txt'):
        labels = load_labels(str(lbl_file))
        for lbl in labels:
            class_counts[lbl['cls']] += 1
    
    class_names = [
        "HM", "LQS", "QHS", "MS",
        "SU-35", "C-130", "C-17", "C-5", "F-16", "TU-160",
        "E-3", "B-52", "P-3C", "B-1B", "E-8",
        "TU-22", "F-15", "KC-135", "F-22", "FA-18",
        "TU-95", "KC-10", "SU-34", "SU-24", "FSC"
    ]
    for cls_id in sorted(class_counts.keys()):
        print(f"  {cls_id:2d} {class_names[cls_id]:<10} {class_counts[cls_id]}")

if __name__ == '__main__':
    main()
