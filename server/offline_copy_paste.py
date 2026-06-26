#!/usr/bin/env python3
"""
离线 Copy-Paste 增强 - 提取尾类目标粘贴到训练图
运行: python offline_copy_paste.py --input images/train --output images/train_cp
"""
import os
import cv2
import random
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def load_labels(lbl_path):
    labels = []
    if not os.path.exists(lbl_path):
        return labels
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-img', type=str, required=True,
                        help='输入图像目录')
    parser.add_argument('--input-lbl', type=str, required=True,
                        help='输入标签目录')
    parser.add_argument('--output-img', type=str, required=True,
                        help='输出图像目录')
    parser.add_argument('--output-lbl', type=str, required=True,
                        help='输出标签目录')
    parser.add_argument('--tail-classes', type=str, default='0,1,21,14,7,24',
                        help='尾类ID列表(逗号分隔)')
    parser.add_argument('--paste-counts', type=str, default='30,25,15,10,8,8',
                        help='每类粘贴数量(逗号分隔)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    tail_classes = [int(x) for x in args.tail_classes.split(',')]
    paste_counts = [int(x) for x in args.paste_counts.split(',')]
    TAIL_CLASSES = dict(zip(tail_classes, paste_counts))

    print(f"尾类配置: {TAIL_CLASSES}")

    os.makedirs(args.output_img, exist_ok=True)
    os.makedirs(args.output_lbl, exist_ok=True)
    random.seed(args.seed)

    # Step 1: 提取尾类目标
    print('Step 1: 提取尾类目标...')
    objects = defaultdict(list)
    img_files = sorted(Path(args.input_img).glob('*.jpg'))
    if not img_files:
        img_files = sorted(Path(args.input_img).glob('*.png'))

    for img_file in tqdm(img_files, desc='Extracting'):
        lbl_file = Path(args.input_lbl) / (img_file.stem + '.txt')
        labels = load_labels(str(lbl_file))
        if not labels:
            continue

        try:
            img = cv2.imdecode(np.fromfile(str(img_file), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
        except Exception:
            continue

        h, w = img.shape[:2]
        for lbl in labels:
            if lbl['cls'] in TAIL_CLASSES:
                bw = lbl['w'] * w
                bh = lbl['h'] * h
                pad = max(5, int(min(bw, bh) * 0.1))
                x1 = max(0, int((lbl['cx'] - lbl['w']/2) * w) - pad)
                y1 = max(0, int((lbl['cy'] - lbl['h']/2) * h) - pad)
                x2 = min(w, int((lbl['cx'] + lbl['w']/2) * w) + pad)
                y2 = min(h, int((lbl['cy'] + lbl['h']/2) * h) + pad)
                crop = img[y1:y2, x1:x2].copy()
                if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
                    objects[lbl['cls']].append(crop)

    for cls_id in sorted(objects.keys()):
        print(f'  Class {cls_id}: {len(objects[cls_id])} objects')

    if sum(len(v) for v in objects.values()) == 0:
        print('错误: 未找到任何尾类目标!')
        sys.exit(1)

    # Step 2: 粘贴
    print('\nStep 2: 粘贴到训练图...')
    augmented = 0
    class_counts = defaultdict(int)

    for img_file in tqdm(img_files, desc='Pasting'):
        lbl_file = Path(args.input_lbl) / (img_file.stem + '.txt')
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

                    # Alpha blending
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
                    class_counts[cls_id] += 1

        # 保存
        out_img = Path(args.output_img) / img_file.name
        out_lbl = Path(args.output_lbl) / (img_file.stem + '.txt')

        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        buf.tofile(str(out_img))

        with open(out_lbl, 'w') as f:
            for lbl in new_labels:
                f.write(f"{lbl['cls']} {lbl['cx']:.6f} {lbl['cy']:.6f} {lbl['w']:.6f} {lbl['h']:.6f}\n")

        augmented += 1

    print(f'\n完成! 增强 {augmented} 张图像')
    print('\n各类新增框数:')
    for c in sorted(class_counts.keys()):
        print(f'  {c}: {class_counts[c]}')


if __name__ == '__main__':
    main()
