"""
SAHI-based Slicing Inference for 10000x10000 Remote Sensing Images

Handles:
1. Large image slicing (640x640 with overlap)
2. Per-slice detection with YOLO
3. Cross-slice NMS and coordinate mapping
4. Inference time optimization (batch inference, sparse inference)
5. Result export in COCO JSON format
"""
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
from PIL import Image

# SAHI imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.postprocess.combine import nms_postprocess
from sahi.models.yolov8 import Yolov8DetectionModel

# COCO category names
COCO_CATEGORIES = [
    {"id": i, "name": name} for i, name in enumerate([
        "HM", "LQS", "QHS", "MS",
        "A1_SU-35", "A2_C-130", "A3_C-17", "A4_C-5", "A5_F-16", "A6_TU-160",
        "A7_E-3", "A8_B-52", "A9_P-3C", "A10_B-1B", "A11_E-8",
        "A12_TU-22", "A13_F-15", "A14_KC-135", "A15_F-22", "A16_FA-18",
        "A17_TU-95", "A18_KC-10", "A19_SU-34", "A20_SU-24", "FSC"
    ])
]


class SahiInferencer:
    """
    SAHI-based inference pipeline for large remote sensing images.
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        slice_size: int = 640,
        overlap_ratio: float = 0.2,
        device: str = "cuda:0",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.device = device
        
        # Load model
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=conf_threshold,
            device=device,
        )
        
        print(f"SAHI Inferencer loaded:")
        print(f"  Model: {model_path}")
        print(f"  Slice: {slice_size}x{slice_size}, overlap: {overlap_ratio}")
        print(f"  Conf: {conf_threshold}, IoU: {iou_threshold}")
    
    def predict_single(
        self,
        image_path: str,
        verbose: bool = True
    ) -> Dict:
        """
        Run sliced prediction on a single image.
        
        Returns:
            Dict with 'predictions', 'inference_time', 'image_size'
        """
        start_time = time.time()
        
        # Get image size
        with Image.open(image_path) as img:
            img_w, img_h = img.size
        
        # Run sliced prediction
        result = get_sliced_prediction(
            image=image_path,
            detection_model=self.detection_model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            postprocess_type="NMS",
            postprocess_match_threshold=self.iou_threshold,
            verbose=0 if not verbose else 1,
        )
        
        inference_time = time.time() - start_time
        
        # Convert to COCO format
        predictions = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox  # [x_min, y_min, width, height]
            predictions.append({
                "category_id": pred.category.id,
                "category_name": pred.category.name,
                "bbox": [bbox.minx, bbox.miny, bbox.maxx - bbox.minx, bbox.maxy - bbox.miny],
                "bbox_yolo": self._to_yolo(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, img_w, img_h),
                "score": pred.score.value,
            })
        
        return {
            "predictions": predictions,
            "inference_time": inference_time,
            "image_size": (img_w, img_h),
            "num_predictions": len(predictions),
        }
    
    def predict_batch(
        self,
        image_paths: List[str],
        output_json: Optional[str] = None,
    ) -> List[Dict]:
        """
        Run prediction on a batch of images.
        
        Args:
            image_paths: List of image file paths
            output_json: Optional path to save COCO JSON results
        Returns:
            List of prediction dicts
        """
        results = []
        total_time = 0
        
        for i, img_path in enumerate(image_paths):
            print(f"[{i+1}/{len(image_paths)}] Processing {Path(img_path).name}...")
            result = self.predict_single(img_path, verbose=False)
            results.append({
                "image_id": i,
                "image_name": Path(img_path).name,
                **result,
            })
            total_time += result["inference_time"]
            print(f"  Time: {result['inference_time']:.2f}s, "
                  f"Predictions: {result['num_predictions']}")
        
        avg_time = total_time / len(image_paths) if image_paths else 0
        print(f"\nTotal: {len(image_paths)} images, {total_time:.2f}s, avg: {avg_time:.2f}s/image")
        
        # Save COCO JSON if requested
        if output_json:
            self._save_coco_json(results, output_json)
            print(f"Results saved to: {output_json}")
        
        return results
    
    def _to_yolo(self, x_min, y_min, x_max, y_max, img_w, img_h):
        """Convert xyxy to YOLO format (normalized xywh)."""
        cx = (x_min + x_max) / 2 / img_w
        cy = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        return [cx, cy, w, h]
    
    def _save_coco_json(self, results: List[Dict], output_path: str):
        """Save results in COCO JSON format."""
        coco_output = {
            "info": {
                "description": "XH-202625 Competition Predictions",
                "version": "1.0",
            },
            "categories": COCO_CATEGORIES,
            "images": [],
            "annotations": [],
        }
        
        ann_id = 1
        for result in results:
            image_id = result["image_id"]
            img_w, img_h = result["image_size"]
            
            coco_output["images"].append({
                "id": image_id,
                "file_name": result["image_name"],
                "width": img_w,
                "height": img_h,
            })
            
            for pred in result["predictions"]:
                coco_output["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": pred["category_id"],
                    "bbox": pred["bbox"],
                    "area": pred["bbox"][2] * pred["bbox"][3],
                    "score": pred["score"],
                    "iscrowd": 0,
                })
                ann_id += 1
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=2)


def evaluate_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """
    Evaluate recall and false alarm rate.
    
    recall = TP / (TP + FN) >= 0.85
    false_alarm_rate = FP / (TP + FP) <= 0.20
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred.get("predictions", [])
        gt_boxes = gt.get("annotations", [])
        
        matched = set()
        for pb in pred_boxes:
            best_iou = 0
            best_gt = -1
            for j, gb in enumerate(gt_boxes):
                if j in matched:
                    continue
                if pb["category_id"] != gb["category_id"]:
                    continue
                iou = compute_iou(pb["bbox"], gb["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = j
            
            if best_iou >= iou_threshold and best_gt >= 0:
                total_tp += 1
                matched.add(best_gt)
            else:
                total_fp += 1
        
        total_fn += len(gt_boxes) - len(matched)
    
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    false_alarm = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    
    return {
        "recall": recall,
        "false_alarm_rate": false_alarm,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "meets_recall": recall >= 0.85,
        "meets_false_alarm": false_alarm <= 0.20,
    }


def compute_iou(box1, box2) -> float:
    """Compute IoU between two boxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    
    if xb <= xa or yb <= ya:
        return 0.0
    
    inter = (xb - xa) * (yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAHI Inference for XH-202625")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--images", type=str, required=True, help="Path to image or directory")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output JSON path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--slice-size", type=int, default=640, help="Slice size")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio")
    
    args = parser.parse_args()
    
    inferencer = SahiInferencer(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap,
    )
    
    # Get image list
    img_path = Path(args.images)
    if img_path.is_dir():
        image_paths = list(img_path.glob("*.jpg")) + list(img_path.glob("*.png"))
    else:
        image_paths = [str(img_path)]
    
    results = inferencer.predict_batch(image_paths, output_json=args.output)
