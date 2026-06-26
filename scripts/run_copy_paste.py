#!/usr/bin/env python3
"""离线 Copy-Paste 增强：提取尾类目标并粘贴到训练图上"""
import os
import cv2
import random
from pathlib import Path
from collections import defaultdict
import numpy as np

TRAIN_IMG_DIR = r'C:\Users\lenovo\Desktop\数据集\data\images\train'
TRAIN_LBL_DIR = r'C:\Users\lenovo\Desktop\数据集\data\labels\train'
OUTPUT_IMG_DIR = r'C:\Users\lenovo\Desktop\数据集\data\images\train_cp'
OUTPUT_LBL_DIR = r'C:\Users\lenovo\Desktop\数据集\data\labels\train_cp'

TAIL_CLASSES = {0: 30, 1: 25, 21: 15, 14: 10, 7: 8, 24: 8}
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)


def load_labels(lbl_path):
    if not os.path.exists(lbl_path):
        return []
    labels = []
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append({
                    'cls': int(parts[0]),
                    'cx': float(parts[1]),
                    'cy': float(parts[2]),
                    'w': float(parts[3]),
                    'h': float(parts[4])
                })
    return labels


def main():
    # Step 1: 提取尾类目标
    print('Step 1: Extracting tail class objects...', flush=True)
    objects = defaultdict(list)
    img_files = sorted(Path(TRAIN_IMG_DIR).glob('*.jpg'))
    print(f'Total images: {len(img_files)}', flush=True)

    skipped = 0
    for i, img_file in enumerate(img_files):
        if i % 500 == 0:
            print(f'  Processing {i}/{len(img_files)}...', flush=True)
        lbl_file = Path(TRAIN_LBL_DIR) / (img_file.stem + '.txt')
        labels = load_labels(str(lbl_file))
        if not labels:
            continue

        try:
            img = cv2.imdecode(np.fromfile(str(img_file), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        h, w = img.shape[:2]
        for lbl in labels:
            if lbl['cls'] in TAIL_CLASSES:
                bw = lbl['w'] * w
                bh = lbl['h'] * h
                x1 = max(0, int((lbl['cx'] - lbl['w']/2) * w) - int(bw * 0.1))
                y1 = max(0, int((lbl['cy'] - lbl['h']/2) * h) - int(bh * 0.1))
                x2 = min(w, int((lbl['cx'] + lbl['w']/2) * w) + int(bw * 0.1))
                y2 = min(h, int((lbl['cy'] + lbl['h']/2) * h) + int(bh * 0.1))
                crop = img[y1:y2, x1:x2].copy()
                if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
                    objects[lbl['cls']].append(crop)

    print(f'Skipped unreadable: {skipped}', flush=True)
    for cls_id in sorted(objects.keys()):
        print(f'  Class {cls_id}: {len(objects[cls_id])} objects', flush=True)

    if sum(len(v) for v in objects.values()) == 0:
        print('FATAL: No tail class objects found!', flush=True)
        return

    # Step 2: 粘贴到训练图
    print('\nStep 2: Pasting onto training images...', flush=True)
    random.seed(42)
    augmented = 0

    for i, img_file in enumerate(img_files):
        if i % 500 == 0:
            print(f'  Pasting {i}/{len(img_files)}...', flush=True)

        lbl_file = Path(TRAIN_LBL_DIR) / (img_file.stem + '.txt')
        labels = load_labels(str(lbl_file))

        try:
            img = cv2.imdecode(np.fromfile(str(img_file), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
        except Exception:
            continue

        h, w = img.shape[:2]
        new_labels = list(labels)

        for cls_id, num_to_paste in TAIL_CLASSES.items():
            if cls_id not in objects or not objects[cls_id]:
                continue
            for _ in range(num_to_paste):
                if random.random() < 0.5:
                    obj = random.choice(objects[cls_id])
                    oh, ow = obj.shape[:2]
                    scale = random.uniform(0.7, 1.3)
                    nw, nh = int(ow * scale), int(oh * scale)
                    if nw >= w or nh >= h or nw < 10 or nh < 10:
                        continue
                    resized = cv2.resize(obj, (nw, nh))
                    px = random.randint(0, w - nw)
                    py = random.randint(0, h - nh)

                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    roi = img[py:py+nh, px:px+nw]
                    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    fg = cv2.bitwise_and(resized, resized, mask=mask)
                    img[py:py+nh, px:px+nw] = cv2.add(bg, fg)

                    cx = (px + nw / 2) / w
                    cy = (py + nh / 2) / h
                    new_labels.append({
                        'cls': cls_id, 'cx': cx, 'cy': cy,
                        'w': nw / w, 'h': nh / h
                    })

        out_img = Path(OUTPUT_IMG_DIR) / img_file.name
        out_lbl = Path(OUTPUT_LBL_DIR) / (img_file.stem + '.txt')

        _, buf = cv2.imencode('.jpg', img)
        buf.tofile(str(out_img))

        with open(out_lbl, 'w') as f:
            for lbl in new_labels:
                f.write(f"{lbl['cls']} {lbl['cx']:.6f} {lbl['cy']:.6f} {lbl['w']:.6f} {lbl['h']:.6f}\n")

        augmented += 1

    print(f'\nDone! Augmented {augmented} images', flush=True)

    # 统计
    print('\nClass distribution:', flush=True)
    class_counts = defaultdict(int)
    for lbl_file in Path(OUTPUT_LBL_DIR).glob('*.txt'):
        for lbl in load_labels(str(lbl_file)):
            class_counts[lbl['cls']] += 1

    names = ['HM','LQS','QHS','MS','SU-35','C-130','C-17','C-5','F-16',
             'TU-160','E-3','B-52','P-3C','B-1B','E-8','TU-22','F-15',
             'KC-135','F-22','FA-18','TU-95','KC-10','SU-34','SU-24','FSC']
    total = 0
    for c in sorted(class_counts.keys()):
        print(f'  {c:2d} {names[c]:<10} {class_counts[c]}')
        total += class_counts[c]
    print(f'  Total: {total}')


if __name__ == '__main__':
    main()
