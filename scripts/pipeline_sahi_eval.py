"""
SAHI 大图切片推理 + 每类阈值优化 + 评估流水线
支持 10000×10000 大图，切片推理后拼接结果
用法: python scripts/pipeline_sahi_eval.py --weights runs/stage1_final/weights/best.pt --mode eval
      python scripts/pipeline_sahi_eval.py --weights runs/stage1_final/weights/best.pt --mode infer --input test_images/ --output results/
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

NAMES = ['HM','LQS','QHS','MS','SU-35','C-130','C-17','C-5','F-16',
         'TU-160','E-3','B-52','P-3C','B-1B','E-8','TU-22','F-15',
         'KC-135','F-22','FA-18','TU-95','KC-10','SU-34','SU-24','FSC']

VAL_IMG_DIR = r"C:\Users\lenovo\Desktop\数据集\data\images\val"
VAL_LBL_DIR = r"C:\Users\lenovo\Desktop\数据集\data\labels\val"


def run_sahi_inference(model_path, img_path, slice_size=640, overlap=0.2, conf=0.01, iou=0.5):
    """SAHI切片推理单张大图"""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel(
        model_type='ultralytics',
        model_path=model_path,
        confidence_threshold=conf,
        device='cuda:0',
    )

    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type='NMS',
        postprocess_match_metric='IOU',
        postprocess_match_threshold=iou,
        verbose=0,
    )

    predictions = []
    for ann in result.object_prediction_list:
        bbox = ann.bbox
        predictions.append({
            'cls': ann.category.id,
            'conf': ann.score.value,
            'box': [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy]
        })
    return predictions


def run_standard_inference(model, img_dir, conf=0.01, iou=0.5, imgsz=640):
    """标准推理（非SAHI），用于验证集评估"""
    print(f"Standard inference on {img_dir}...")
    results = model.predict(
        source=str(img_dir),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=0,
        verbose=False,
        save=False,
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

    print(f"  Inference done: {len(all_preds)} images")
    return all_preds


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


def load_gt_labels(img_dir, lbl_dir):
    """加载所有GT标签"""
    gt_data = {}
    img_files = sorted(Path(img_dir).glob('*.jpg'))
    for img_file in img_files:
        lbl_path = Path(lbl_dir) / (img_file.stem + '.txt')
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
            with Image.open(str(img_file)) as im:
                img_w, img_h = im.size
            for lbl in labels:
                lbl['box'] = [
                    (lbl['cx'] - lbl['w'] / 2) * img_w,
                    (lbl['cy'] - lbl['h'] / 2) * img_h,
                    (lbl['cx'] + lbl['w'] / 2) * img_w,
                    (lbl['cy'] + lbl['h'] / 2) * img_h
                ]
        gt_data[img_file.name] = labels
    return gt_data


def evaluate(all_preds, gt_data, iou_thresh=0.5, per_class_conf=None):
    """评估，支持每类不同置信度阈值"""
    cls_gt_count = defaultdict(int)
    for img_name, gts in gt_data.items():
        for gt in gts:
            cls_gt_count[gt['cls']] += 1

    # 按类别收集检测结果
    cls_detections = defaultdict(list)

    for img_name, preds in all_preds.items():
        gts = gt_data.get(img_name, [])
        matched = [False] * len(gts)

        preds_sorted = sorted(preds, key=lambda x: -x['conf'])
        for pred in preds_sorted:
            pred_cls = pred['cls']
            pred_box = pred['box']
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(gts):
                if matched[j] or gt['cls'] != pred_cls:
                    continue
                cur_iou = iou(pred_box, gt['box'])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_gt_idx = j

            is_tp = best_iou >= iou_thresh and best_gt_idx >= 0
            if is_tp:
                matched[best_gt_idx] = True
            cls_detections[pred_cls].append((pred['conf'], is_tp))

    # 计算每类指标
    cls_metrics = {}
    total_tp = total_fp = total_fn = 0

    for cls_id in range(25):
        dets = cls_detections.get(cls_id, [])
        gt_count = cls_gt_count.get(cls_id, 0)
        conf_thresh = per_class_conf.get(cls_id, 0.25) if per_class_conf else 0.25

        tp = sum(1 for conf, is_tp in dets if conf >= conf_thresh and is_tp)
        fp = sum(1 for conf, is_tp in dets if conf >= conf_thresh and not is_tp)
        fn = gt_count - tp

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

        cls_metrics[cls_id] = {
            'name': NAMES[cls_id], 'threshold': conf_thresh,
            'recall': recall, 'precision': precision, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'gt_count': gt_count
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    global_metrics = {
        'recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
        'precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'far': total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'f1': 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn
    }

    return cls_metrics, global_metrics


def optimize_per_class(all_preds, gt_data, iou_thresh=0.5):
    """扫描每类最优阈值"""
    thresholds = np.arange(0.05, 0.55, 0.05)

    cls_gt_count = defaultdict(int)
    for img_name, gts in gt_data.items():
        for gt in gts:
            cls_gt_count[gt['cls']] += 1

    cls_detections = defaultdict(list)
    for img_name, preds in all_preds.items():
        gts = gt_data.get(img_name, [])
        matched = [False] * len(gts)
        preds_sorted = sorted(preds, key=lambda x: -x['conf'])
        for pred in preds_sorted:
            pred_cls = pred['cls']
            best_iou = 0
            best_gt_idx = -1
            for j, gt in enumerate(gts):
                if matched[j] or gt['cls'] != pred_cls:
                    continue
                cur_iou = iou(pred['box'], gt['box'])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_gt_idx = j
            is_tp = best_iou >= iou_thresh and best_gt_idx >= 0
            if is_tp:
                matched[best_gt_idx] = True
            cls_detections[pred_cls].append((pred['conf'], is_tp))

    optimal = {}
    for cls_id in range(25):
        dets = cls_detections.get(cls_id, [])
        gt_count = cls_gt_count.get(cls_id, 0)
        if gt_count == 0:
            optimal[cls_id] = 0.25
            continue

        best_f1 = -1
        best_thresh = 0.25
        for thresh in thresholds:
            tp = sum(1 for c, t in dets if c >= thresh and t)
            fp = sum(1 for c, t in dets if c >= thresh and not t)
            fn = gt_count - tp
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        optimal[cls_id] = float(best_thresh)

    return optimal


def mode_eval(args):
    """评估模式：在验证集上评估"""
    from ultralytics import YOLO

    print("=" * 70)
    print("评估模式: 验证集")
    print(f"权重: {args.weights}")
    print("=" * 70)

    model = YOLO(args.weights)

    # 推理
    all_preds = run_standard_inference(model, VAL_IMG_DIR, conf=0.01, iou=0.5)

    # 加载GT
    gt_data = load_gt_labels(VAL_IMG_DIR, VAL_LBL_DIR)
    print(f"GT: {len(gt_data)} images, {sum(len(v) for v in gt_data.values())} boxes")

    # 优化每类阈值
    print("\n优化每类阈值...")
    optimal = optimize_per_class(all_preds, gt_data, iou_thresh=0.5)

    # 用最优阈值评估
    cls_metrics, global_metrics = evaluate(all_preds, gt_data, iou_thresh=0.5, per_class_conf=optimal)

    # 打印
    print("\n" + "=" * 70)
    print("每类评估结果 (最优阈值):")
    print(f"{'Class':<15} {'Thresh':>7} {'Recall':>8} {'Prec':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'GT':>6}")
    print("-" * 85)
    for cls_id in range(25):
        m = cls_metrics[cls_id]
        print(f"{m['name']:<15} {m['threshold']:>7.2f} {m['recall']:>8.4f} {m['precision']:>8.4f} {m['f1']:>8.4f} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} {m['gt_count']:>6}")

    print("\n" + "=" * 70)
    print("全局指标:")
    print(f"  召回率 (Recall): {global_metrics['recall']:.4f}  (目标 ≥ 0.85)")
    print(f"  虚警率 (FAR):    {global_metrics['far']:.4f}  (目标 ≤ 0.20)")
    print(f"  精确率 (Precision): {global_metrics['precision']:.4f}")
    print(f"  F1-score:        {global_metrics['f1']:.4f}")
    print(f"  TP={global_metrics['tp']}, FP={global_metrics['fp']}, FN={global_metrics['fn']}")

    # 保存
    output = {
        'optimal_thresholds': {NAMES[k]: v for k, v in optimal.items()},
        'per_class': {NAMES[k]: v for k, v in cls_metrics.items()},
        'global': global_metrics,
    }
    out_path = Path(args.weights).parent / 'eval_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果: {out_path}")


def mode_infer(args):
    """推理模式：SAHI切片推理大图"""
    print("=" * 70)
    print("推理模式: SAHI切片推理")
    print(f"权重: {args.weights}")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print("=" * 70)

    # 加载每类阈值
    thresh_file = Path(args.weights).parent / 'best_thresholds.json'
    per_class_conf = {}
    if thresh_file.exists():
        with open(thresh_file, 'r') as f:
            thresh_dict = json.load(f)
        per_class_conf = {NAMES.index(k): v for k, v in thresh_dict.items()}
        print(f"加载每类阈值: {thresh_file}")
    else:
        print("未找到阈值文件，使用默认0.25")

    os.makedirs(args.output, exist_ok=True)
    img_files = sorted(Path(args.input).glob('*.jpg')) + sorted(Path(args.input).glob('*.png'))
    if not img_files:
        img_files = sorted(Path(args.input).glob('*.tif')) + sorted(Path(args.input).glob('*.tiff'))

    print(f"找到 {len(img_files)} 张图片")

    all_coco_results = []
    image_id = 0

    for img_file in img_files:
        t0 = time.time()
        print(f"\n推理: {img_file.name}")

        preds = run_sahi_inference(
            args.weights, str(img_file),
            slice_size=args.slice_size, overlap=args.overlap,
            conf=0.01, iou=0.5
        )

        # 应用每类阈值
        filtered = []
        for p in preds:
            thresh = per_class_conf.get(p['cls'], 0.25)
            if p['conf'] >= thresh:
                filtered.append(p)

        t1 = time.time()
        print(f"  切片推理: {t1-t0:.1f}s, 过滤后: {len(filtered)} 检测")

        # 保存可视化
        if args.save_vis:
            import cv2
            img = cv2.imdecode(np.fromfile(str(img_file), dtype=np.uint8), cv2.IMREAD_COLOR)
            for p in filtered:
                x1, y1, x2, y2 = map(int, p['box'])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{NAMES[p['cls']]} {p['conf']:.2f}"
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            vis_path = Path(args.output) / (img_file.stem + '_vis.jpg')
            _, buf = cv2.imencode('.jpg', img)
            buf.tofile(str(vis_path))

        # COCO格式
        for p in filtered:
            x1, y1, x2, y2 = p['box']
            all_coco_results.append({
                'image_id': image_id,
                'category_id': p['cls'],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': p['conf'],
                'category_name': NAMES[p['cls']]
            })

        image_id += 1

    # 保存COCO JSON
    coco_path = Path(args.output) / 'predictions.json'
    with open(coco_path, 'w', encoding='utf-8') as f:
        json.dump(all_coco_results, f, indent=2)
    print(f"\nCOCO结果: {coco_path}")
    print(f"总检测: {len(all_coco_results)}")


def main():
    parser = argparse.ArgumentParser(description='SAHI推理+评估流水线')
    parser.add_argument('--weights', default='runs/stage1_final/weights/best.pt')
    parser.add_argument('--mode', choices=['eval', 'infer'], default='eval')
    parser.add_argument('--input', default=None, help='推理模式输入目录')
    parser.add_argument('--output', default='results/', help='推理模式输出目录')
    parser.add_argument('--slice_size', type=int, default=640)
    parser.add_argument('--overlap', type=float, default=0.2)
    parser.add_argument('--save_vis', action='store_true')
    args = parser.parse_args()

    if args.mode == 'eval':
        mode_eval(args)
    else:
        if not args.input:
            print("推理模式需要 --input 参数")
            sys.exit(1)
        mode_infer(args)


if __name__ == '__main__':
    main()
