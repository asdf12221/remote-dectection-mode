#!/usr/bin/env python3
"""
每类阈值优化 + 评估脚本
在验证集上扫描不同置信度阈值，找到使F1-score最优的每类阈值
运行: python optimize_thresholds.py --weights runs/stage1/weights/best.pt --data dataset.yaml
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from ultralytics import YOLO

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

NAMES = ['HM','LQS','QHS','MS','SU-35','C-130','C-17','C-5','F-16',
         'TU-160','E-3','B-52','P-3C','B-1B','E-8','TU-22','F-15',
         'KC-135','F-22','FA-18','TU-95','KC-10','SU-34','SU-24','FSC']

THRESHOLDS = np.arange(0.05, 0.55, 0.05)


def load_yaml(yaml_path):
    """简单解析YAML获取数据集路径"""
    import yaml as yL
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yL.safe_load(f)
    base = data.get('path', '.')
    val_img = os.path.join(base, data.get('val', 'images/val'))
    val_lbl = os.path.join(base, 'labels', os.path.basename(data.get('val', 'images/val').replace('images/', '')))
    # 标签目录推断: images/val -> labels/val
    if 'images' in val_img:
        val_lbl = val_img.replace('images', 'labels')
    return val_img, val_lbl, data.get('names', {})


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def load_gt_labels(val_img_dir, val_lbl_dir):
    gt_data = {}
    img_files = sorted(Path(val_img_dir).glob('*.jpg'))
    if not img_files:
        img_files = sorted(Path(val_img_dir).glob('*.png'))
    for img_file in img_files:
        lbl_path = Path(val_lbl_dir) / (img_file.stem + '.txt')
        labels = []
        if os.path.exists(lbl_path):
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
        if labels:
            try:
                with Image.open(str(img_file)) as im:
                    img_w, img_h = im.size
            except Exception:
                continue
            for lbl in labels:
                lbl['box'] = [
                    (lbl['cx'] - lbl['w'] / 2) * img_w,
                    (lbl['cy'] - lbl['h'] / 2) * img_h,
                    (lbl['cx'] + lbl['w'] / 2) * img_w,
                    (lbl['cy'] + lbl['h'] / 2) * img_h
                ]
        gt_data[img_file.name] = labels
    return gt_data


def run_inference(model, img_dir, conf=0.01, iou=0.5, imgsz=640):
    print(f"推理中 (conf={conf}, iou={iou}, imgsz={imgsz})...")
    results = model.predict(
        source=str(img_dir),
        conf=conf, iou=iou, imgsz=imgsz,
        device=0, verbose=False, save=False,
    )
    all_preds = {}
    for r in results:
        img_name = Path(r.path).name
        preds = []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(confs)):
                preds.append({
                    'cls': int(clses[i]),
                    'conf': float(confs[i]),
                    'box': [float(x) for x in boxes[i]]
                })
        all_preds[img_name] = preds
    print(f"推理完成: {len(all_preds)} 张图")
    return all_preds


def optimize_and_evaluate(all_preds, gt_data, iou_thresh=0.5):
    """扫描每类最优阈值并计算指标"""
    cls_gt_count = defaultdict(int)
    for img_name, gts in gt_data.items():
        for gt in gts:
            cls_gt_count[gt['cls']] += 1

    # 匹配
    cls_detections = defaultdict(list)
    for img_name, preds in all_preds.items():
        gts = gt_data.get(img_name, [])
        matched = [False] * len(gts)
        for pred in sorted(preds, key=lambda x: -x['conf']):
            best_iou = 0
            best_idx = -1
            for j, gt in enumerate(gts):
                if matched[j] or gt['cls'] != pred['cls']:
                    continue
                cur = iou(pred['box'], gt['box'])
                if cur > best_iou:
                    best_iou = cur
                    best_idx = j
            is_tp = best_iou >= iou_thresh and best_idx >= 0
            if is_tp:
                matched[best_idx] = True
            cls_detections[pred['cls']].append((pred['conf'], is_tp))

    # 每类最优阈值
    optimal = {}
    cls_results = {}
    for cls_id in range(25):
        dets = cls_detections.get(cls_id, [])
        gt_count = cls_gt_count.get(cls_id, 0)
        if gt_count == 0:
            optimal[cls_id] = 0.25
            cls_results[cls_id] = {'threshold': 0.25, 'recall': 0, 'precision': 0, 'f1': 0,
                                   'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0}
            continue

        best_f1 = -1
        best_thresh = 0.25
        best_metrics = {}
        for thresh in THRESHOLDS:
            tp = sum(1 for c, t in dets if c >= thresh and t)
            fp = sum(1 for c, t in dets if c >= thresh and not t)
            fn = gt_count - tp
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
                best_metrics = {'threshold': best_thresh, 'recall': recall,
                                'precision': precision, 'f1': f1,
                                'tp': tp, 'fp': fp, 'fn': fn, 'gt': gt_count}
        optimal[cls_id] = best_thresh
        cls_results[cls_id] = best_metrics

    # 全局指标 (每类最优阈值)
    total_tp = total_fp = total_fn = 0
    for cls_id in range(25):
        dets = cls_detections.get(cls_id, [])
        gt_count = cls_gt_count.get(cls_id, 0)
        thresh = optimal[cls_id]
        tp = sum(1 for c, t in dets if c >= thresh and t)
        fp = sum(1 for c, t in dets if c >= thresh and not t)
        fn = gt_count - tp
        total_tp += tp; total_fp += fp; total_fn += fn

    global_metrics = {
        'recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
        'precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'far': total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'f1': 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn
    }

    # 全局统一阈值
    global_by_thresh = {}
    for thresh in THRESHOLDS:
        tp = fp = fn = 0
        for cls_id in range(25):
            dets = cls_detections.get(cls_id, [])
            gt_count = cls_gt_count.get(cls_id, 0)
            tp += sum(1 for c, t in dets if c >= thresh and t)
            fp += sum(1 for c, t in dets if c >= thresh and not t)
            fn += gt_count - sum(1 for c, t in dets if c >= thresh and t)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        global_by_thresh[float(thresh)] = {'recall': recall, 'precision': precision,
                                            'far': far, 'f1': f1}

    return optimal, cls_results, global_metrics, global_by_thresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='模型权重路径')
    parser.add_argument('--data', default='dataset.yaml', help='数据集YAML')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--output', default=None, help='输出JSON路径')
    args = parser.parse_args()

    val_img_dir, val_lbl_dir, names = load_yaml(args.data)
    print(f"验证集图像: {val_img_dir}")
    print(f"验证集标签: {val_lbl_dir}")

    if not os.path.isdir(val_img_dir):
        print(f"错误: 验证集目录不存在: {val_img_dir}")
        sys.exit(1)

    model = YOLO(args.weights)
    gt_data = load_gt_labels(val_img_dir, val_lbl_dir)
    print(f"GT: {len(gt_data)} 图, {sum(len(v) for v in gt_data.values())} 框")

    all_preds = run_inference(model, val_img_dir, conf=0.01, iou=0.5, imgsz=args.imgsz)

    optimal, cls_results, global_metrics, global_by_thresh = optimize_and_evaluate(
        all_preds, gt_data, iou_thresh=0.5)

    # 打印
    print("\n" + "=" * 85)
    print("每类最优阈值:")
    print(f"{'Class':<12} {'Thresh':>7} {'Recall':>8} {'Prec':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'GT':>6}")
    print("-" * 85)
    for cls_id in range(25):
        r = cls_results[cls_id]
        print(f"{NAMES[cls_id]:<12} {r['threshold']:>7.2f} {r['recall']:>8.4f} {r['precision']:>8.4f} "
              f"{r['f1']:>8.4f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} {r['gt']:>6}")

    print("\n" + "=" * 70)
    print("全局指标 (每类最优阈值):")
    print(f"  召回率 (Recall): {global_metrics['recall']:.4f}  目标 ≥ 0.85")
    print(f"  虚警率 (FAR):    {global_metrics['far']:.4f}  目标 ≤ 0.20")
    print(f"  精确率 (Precision): {global_metrics['precision']:.4f}")
    print(f"  F1:              {global_metrics['f1']:.4f}")
    print(f"  TP={global_metrics['tp']}, FP={global_metrics['fp']}, FN={global_metrics['fn']}")

    print("\n全局统一阈值对比:")
    print(f"{'Thresh':>8} {'Recall':>8} {'Prec':>8} {'FAR':>8} {'F1':>8}")
    for t in sorted(global_by_thresh.keys()):
        r = global_by_thresh[t]
        print(f"{t:>8.2f} {r['recall']:>8.4f} {r['precision']:>8.4f} {r['far']:>8.4f} {r['f1']:>8.4f}")

    # 保存
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.weights), 'eval_results.json')

    output = {
        'optimal_thresholds': {NAMES[k]: v for k, v in optimal.items()},
        'per_class': {NAMES[k]: v for k, v in cls_results.items()},
        'global': global_metrics,
        'global_by_thresh': global_by_thresh,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 保存简单阈值文件
    thresh_file = os.path.join(os.path.dirname(args.output), 'best_thresholds.json')
    with open(thresh_file, 'w', encoding='utf-8') as f:
        json.dump({NAMES[k]: v for k, v in optimal.items()}, f, indent=2)

    print(f"\n结果: {args.output}")
    print(f"阈值: {thresh_file}")


if __name__ == '__main__':
    main()
