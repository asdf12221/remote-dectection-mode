"""
每类置信度阈值优化：在验证集上扫描不同阈值，找到使 F1-score 最优的每类阈值。
目标：召回率 ≥ 85%，虚警率 ≤ 20%
用法: python scripts/optimize_thresholds.py --weights runs/stage1_final/weights/best.pt
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

DATA_YAML = r"C:\Users\lenovo\Desktop\数据集\data\dataset_balanced.yaml"
VAL_IMG_DIR = r"C:\Users\lenovo\Desktop\数据集\data\images\val"
VAL_LBL_DIR = r"C:\Users\lenovo\Desktop\数据集\data\labels\val"

NAMES = ['HM','LQS','QHS','MS','SU-35','C-130','C-17','C-5','F-16',
         'TU-160','E-3','B-52','P-3C','B-1B','E-8','TU-22','F-15',
         'KC-135','F-22','FA-18','TU-95','KC-10','SU-34','SU-24','FSC']

# 候选阈值范围
THRESHOLDS = np.arange(0.05, 0.55, 0.05)


def load_labels(lbl_path):
    """加载YOLO格式标签"""
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


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """YOLO格式转xyxy"""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def iou(box1, box2):
    """计算IoU"""
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


def run_inference(model, img_dir, conf=0.01, iou_thresh=0.5):
    """在验证集上推理，保存所有预测结果"""
    print(f"Running inference with conf={conf}, iou={iou_thresh}...")
    img_files = sorted(Path(img_dir).glob('*.jpg'))
    all_predictions = {}  # img_name -> list of (cls, conf, box)

    results = model.predict(
        source=str(img_dir),
        conf=conf,
        iou=iou_thresh,
        imgsz=640,
        device=0,
        verbose=False,
        save=False,
    )

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
        all_predictions[img_name] = preds

    print(f"Inference done: {len(all_predictions)} images")
    return all_predictions


def evaluate_at_thresholds(all_predictions, gt_data, iou_thresh=0.5):
    """在不同置信度阈值下评估每类和全局指标"""
    # 按类别收集所有 (conf, is_tp) 对
    cls_detections = defaultdict(list)  # cls -> list of (conf, is_tp)

    for img_name, preds in all_predictions.items():
        gts = gt_data.get(img_name, [])
        matched = [False] * len(gts)

        # 按置信度降序排列预测
        preds_sorted = sorted(preds, key=lambda x: -x['conf'])

        for pred in preds_sorted:
            pred_cls = pred['cls']
            pred_box = pred['box']
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(gts):
                if matched[j] or gt['cls'] != pred_cls:
                    continue
                gt_box = gt['box']
                cur_iou = iou(pred_box, gt_box)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_gt_idx = j

            is_tp = best_iou >= iou_thresh and best_gt_idx >= 0
            if is_tp:
                matched[best_gt_idx] = True

            cls_detections[pred_cls].append((pred['conf'], is_tp))

    # 统计每类GT数量
    cls_gt_count = defaultdict(int)
    for img_name, gts in gt_data.items():
        for gt in gts:
            cls_gt_count[gt['cls']] += 1

    # 对每个类别找最优阈值
    optimal_thresholds = {}
    cls_results = {}

    for cls_id in range(25):
        dets = cls_detections.get(cls_id, [])
        gt_count = cls_gt_count.get(cls_id, 0)

        if gt_count == 0:
            optimal_thresholds[cls_id] = 0.25
            cls_results[cls_id] = {'threshold': 0.25, 'recall': 0, 'precision': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0}
            continue

        best_f1 = -1
        best_thresh = 0.25
        best_metrics = {}

        for thresh in THRESHOLDS:
            tp = sum(1 for conf, is_tp in dets if conf >= thresh and is_tp)
            fp = sum(1 for conf, is_tp in dets if conf >= thresh and not is_tp)
            fn = gt_count - tp

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_metrics = {
                    'threshold': float(thresh),
                    'recall': recall,
                    'precision': precision,
                    'f1': f1,
                    'tp': tp, 'fp': fp, 'fn': fn
                }

        optimal_thresholds[cls_id] = best_thresh
        cls_results[cls_id] = best_metrics

    # 计算全局指标
    global_results = {}
    for thresh in THRESHOLDS:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for cls_id in range(25):
            dets = cls_detections.get(cls_id, [])
            gt_count = cls_gt_count.get(cls_id, 0)
            tp = sum(1 for conf, is_tp in dets if conf >= thresh and is_tp)
            fp = sum(1 for conf, is_tp in dets if conf >= thresh and not is_tp)
            fn = gt_count - tp
            total_tp += tp
            total_fp += fp
            total_fn += fn

        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        far = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

        global_results[float(thresh)] = {
            'recall': recall, 'precision': precision, 'far': far, 'f1': f1,
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn
        }

    # 使用每类最优阈值的全局指标
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for cls_id in range(25):
        dets = cls_detections.get(cls_id, [])
        gt_count = cls_gt_count.get(cls_id, 0)
        thresh = optimal_thresholds[cls_id]
        tp = sum(1 for conf, is_tp in dets if conf >= thresh and is_tp)
        fp = sum(1 for conf, is_tp in dets if conf >= thresh and not is_tp)
        fn = gt_count - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    per_class_global = {
        'recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
        'precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'far': total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'f1': 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn
    }

    return optimal_thresholds, cls_results, global_results, per_class_global


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='runs/stage1_final/weights/best.pt',
                        help='模型权重路径')
    parser.add_argument('--output', default='runs/stage1_final/threshold_optimization.json',
                        help='输出JSON路径')
    args = parser.parse_args()

    print("=" * 70)
    print("每类置信度阈值优化")
    print(f"权重: {args.weights}")
    print(f"验证集: {VAL_IMG_DIR}")
    print("=" * 70)

    # 加载模型
    model = YOLO(args.weights)

    # 加载GT
    print("Loading ground truth labels...")
    gt_data = {}
    img_files = sorted(Path(VAL_IMG_DIR).glob('*.jpg'))
    for img_file in img_files:
        lbl_path = Path(VAL_LBL_DIR) / (img_file.stem + '.txt')
        labels = load_labels(str(lbl_path))
        # 获取图像尺寸
        from PIL import Image
        try:
            with Image.open(str(img_file)) as im:
                img_w, img_h = im.size
        except Exception:
            continue

        gt_boxes = []
        for lbl in labels:
            box = yolo_to_xyxy(lbl['cx'], lbl['cy'], lbl['w'], lbl['h'], img_w, img_h)
            gt_boxes.append({'cls': lbl['cls'], 'box': box})
        gt_data[img_file.name] = gt_boxes

    print(f"Loaded {len(gt_data)} images with {sum(len(v) for v in gt_data.values())} GT boxes")

    # 推理
    all_predictions = run_inference(model, VAL_IMG_DIR, conf=0.01, iou_thresh=0.5)

    # 评估
    print("\nOptimizing per-class thresholds...")
    optimal_thresholds, cls_results, global_results, per_class_global = evaluate_at_thresholds(
        all_predictions, gt_data, iou_thresh=0.5
    )

    # 打印结果
    print("\n" + "=" * 70)
    print("每类最优阈值:")
    print(f"{'Class':<15} {'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 80)
    for cls_id in range(25):
        r = cls_results[cls_id]
        print(f"{NAMES[cls_id]:<15} {r['threshold']:>10.2f} {r['recall']:>8.4f} {r['precision']:>10.4f} {r['f1']:>8.4f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6}")

    print("\n" + "=" * 70)
    print("全局指标 (每类最优阈值):")
    print(f"  召回率 (Recall): {per_class_global['recall']:.4f}  (目标 ≥ 0.85)")
    print(f"  虚警率 (FAR):    {per_class_global['far']:.4f}  (目标 ≤ 0.20)")
    print(f"  精确率 (Precision): {per_class_global['precision']:.4f}")
    print(f"  F1-score:        {per_class_global['f1']:.4f}")
    print(f"  TP={per_class_global['tp']}, FP={per_class_global['fp']}, FN={per_class_global['fn']}")

    print("\n全局指标 (统一阈值):")
    print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'FAR':>8} {'F1':>8}")
    print("-" * 50)
    for thresh in sorted(global_results.keys()):
        r = global_results[thresh]
        print(f"{thresh:>10.2f} {r['recall']:>8.4f} {r['precision']:>10.4f} {r['far']:>8.4f} {r['f1']:>8.4f}")

    # 保存结果
    output = {
        'optimal_thresholds': {str(k): v for k, v in optimal_thresholds.items()},
        'per_class_results': {NAMES[k]: v for k, v in cls_results.items()},
        'per_class_global': per_class_global,
        'global_results': {str(k): v for k, v in global_results.items()},
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {args.output}")

    # 保存每类阈值为简单格式供推理使用
    thresh_file = Path(args.output).parent / 'best_thresholds.json'
    simple = {NAMES[k]: v for k, v in optimal_thresholds.items()}
    with open(thresh_file, 'w', encoding='utf-8') as f:
        json.dump(simple, f, indent=2)
    print(f"阈值文件: {thresh_file}")


if __name__ == '__main__':
    main()
