"""
评估脚本 - 在验证集上评估模型，计算召回率和虚警率
用法: python scripts/evaluate.py --model path/to/best.pt
"""
import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from ultralytics import YOLO


def compute_iou(box1, box2):
    """IoU between two boxes in [x1, y1, x2, y2] format."""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def load_yolo_labels(label_path, img_w, img_h):
    """Load YOLO format labels -> list of [class_id, x1, y1, x2, y2]."""
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            boxes.append([cls, x1, y1, x2, y2])
    return boxes


def evaluate_model(
    model_path,
    val_images_dir,
    val_labels_dir,
    conf=0.25,
    iou_threshold=0.5,
    imgsz=512,
):
    """
    Evaluate model on validation set.
    Returns per-class and overall metrics.
    """
    model = YOLO(model_path)
    
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_paths.extend(Path(val_images_dir).glob(f'*{ext}'))
    
    # Per-class counters
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    class_total_gt = defaultdict(int)
    class_total_pred = defaultdict(int)
    
    total_time = 0
    
    for img_path in sorted(image_paths):
        # Get image size
        from PIL import Image
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        
        # Load GT
        label_path = Path(val_labels_dir) / (img_path.stem + '.txt')
        gt_boxes = load_yolo_labels(str(label_path), img_w, img_h)
        for gb in gt_boxes:
            class_total_gt[gb[0]] += 1
        
        # Predict
        t0 = time.time()
        results = model.predict(
            str(img_path),
            conf=conf,
            iou=0.45,
            imgsz=imgsz,
            device=0,
            verbose=False,
        )
        total_time += time.time() - t0
        
        # Parse predictions
        pred_boxes = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                score = boxes.conf[i].item()
                pred_boxes.append([cls, x1, y1, x2, y2, score])
                class_total_pred[cls] += 1
        
        # Match predictions to GT
        matched_gt = set()
        for pb in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                if pb[0] != gb[0]:
                    continue
                iou = compute_iou(pb[1:5], gb[1:5])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                class_tp[pb[0]] += 1
                matched_gt.add(best_gt_idx)
            else:
                class_fp[pb[0]] += 1
        
        for j, gb in enumerate(gt_boxes):
            if j not in matched_gt:
                class_fn[gb[0]] += 1
    
    # Compute metrics
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_far = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    avg_time = total_time / len(image_paths) if image_paths else 0
    
    # Per-class metrics
    per_class = {}
    class_names = [
        "HM", "LQS", "QHS", "MS",
        "A1_SU-35", "A2_C-130", "A3_C-17", "A4_C-5", "A5_F-16", "A6_TU-160",
        "A7_E-3", "A8_B-52", "A9_P-3C", "A10_B-1B", "A11_E-8",
        "A12_TU-22", "A13_F-15", "A14_KC-135", "A15_F-22", "A16_FA-18",
        "A17_TU-95", "A18_KC-10", "A19_SU-34", "A20_SU-24", "FSC"
    ]
    
    for cls_id in range(25):
        tp = class_tp[cls_id]
        fp = class_fp[cls_id]
        fn = class_fn[cls_id]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        per_class[class_names[cls_id]] = {
            "tp": tp, "fp": fp, "fn": fn,
            "gt_count": class_total_gt[cls_id],
            "pred_count": class_total_pred[cls_id],
            "recall": recall,
            "precision": precision,
        }
    
    return {
        "overall": {
            "recall": overall_recall,
            "false_alarm_rate": overall_far,
            "precision": overall_precision,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "meets_recall": overall_recall >= 0.85,
            "meets_false_alarm": overall_far <= 0.20,
            "avg_inference_time": avg_time,
            "num_images": len(image_paths),
        },
        "per_class": per_class,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on validation set")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--val-images", type=str, 
                        default=r"C:\Users\lenovo\Desktop\数据集\data\val\images",
                        help="Validation images directory")
    parser.add_argument("--val-labels", type=str,
                        default=r"C:\Users\lenovo\Desktop\数据集\data\val\labels",
                        help="Validation labels directory")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--output", type=str, default="eval_results.json")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("模型评估")
    print(f"模型: {args.model}")
    print(f"验证集: {args.val_images}")
    print("=" * 70)
    
    results = evaluate_model(
        model_path=args.model,
        val_images_dir=args.val_images,
        val_labels_dir=args.val_labels,
        conf=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
    )
    
    # Print summary
    o = results["overall"]
    print(f"\n{'='*70}")
    print(f"总体指标:")
    print(f"  召回率: {o['recall']:.4f} {'✅' if o['meets_recall'] else '❌'} (目标≥0.85)")
    print(f"  虚警率: {o['false_alarm_rate']:.4f} {'✅' if o['meets_false_alarm'] else '❌'} (目标≤0.20)")
    print(f"  精确率: {o['precision']:.4f}")
    print(f"  TP={o['tp']}, FP={o['fp']}, FN={o['fn']}")
    print(f"  平均推理时间: {o['avg_inference_time']:.3f}s/image")
    print(f"  图片数: {o['num_images']}")
    
    print(f"\n{'='*70}")
    print(f"{'类别':<15} {'GT':>5} {'Pred':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'Recall':>8} {'Prec':>8}")
    print("-" * 70)
    for cls_name, m in sorted(results["per_class"].items()):
        print(f"{cls_name:<15} {m['gt_count']:>5} {m['pred_count']:>5} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {m['recall']:>8.4f} {m['precision']:>8.4f}")
    
    # Save JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.output}")
