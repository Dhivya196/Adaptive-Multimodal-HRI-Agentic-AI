"""
Vision Detection Evaluation Script for COCO 2017 Curated Subset.

Evaluates the Adaptive Multimodal HRI Framework's YOLO vision detector
against ground-truth COCO 2017 bounding boxes.

Calculates valid, rigorous object detection metrics:
- Precision
- Recall
- Mean IoU (Intersection over Union)
- mAP@50 (Mean Average Precision at IoU=0.50)
- mAP@50:95 (Mean Average Precision across IoU=0.50:0.05:0.95)

Classes Evaluated:
- person
- bottle
- cup
- chair
- couch (sofa)
- potted plant
- laptop
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ANNOTATIONS_PATH = Path("data/coco_subset/annotations_subset.json")
IMAGES_DIR = Path("data/coco_subset/images")
MODEL_PATH = Path("yolo26n.pt")

CLASS_NAME_ALIASES = {
    "couch": ["couch", "sofa"],
    "potted plant": ["potted plant", "pottedplant", "plant"],
    "bottle": ["bottle"],
    "cup": ["cup"],
    "laptop": ["laptop"],
    "person": ["person"],
    "chair": ["chair"],
}


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_ap_for_class(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.50,
) -> Tuple[float, float, float, List[float]]:
    """
    Compute Precision, Recall, Average Precision (AP), and matched IoUs for a single class at given IoU threshold.
    """
    total_gts = len(ground_truths)
    if total_gts == 0:
        return 0.0, 0.0, 0.0, []

    if len(predictions) == 0:
        return 0.0, 0.0, 0.0, []

    # Sort predictions by confidence descending
    preds_sorted = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    # Track which GT boxes have been matched per image
    matched_gt_ids = set()
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))
    matched_ious = []

    for idx, pred in enumerate(preds_sorted):
        pred_box = pred["bbox"]
        img_id = pred["image_id"]

        # Find candidate GTs in the same image
        candidate_gts = [gt for gt in ground_truths if gt["image_id"] == img_id]

        best_iou = 0.0
        best_gt_idx = -1

        for gt in candidate_gts:
            iou = compute_iou(pred_box, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt["id"]

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt_ids:
            tp[idx] = 1.0
            matched_gt_ids.add(best_gt_idx)
            matched_ious.append(best_iou)
        else:
            fp[idx] = 1.0

    # Cumulative sums
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    precisions = cum_tp / (cum_tp + cum_fp + 1e-10)
    recalls = cum_tp / (total_gts + 1e-10)

    final_precision = float(precisions[-1]) if len(precisions) > 0 else 0.0
    final_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0

    # 11-point interpolated AP
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        prec_at_rec = precisions[recalls >= t]
        p = float(np.max(prec_at_rec)) if len(prec_at_rec) > 0 else 0.0
        ap += p / 11.0

    return final_precision, final_recall, ap, matched_ious


def evaluate_vision_dataset(annotations_path: Path, images_dir: Path, model_path: Path) -> Dict:
    """Run full object detection evaluation against the curated COCO subset."""
    if not annotations_path.exists():
        print(f"Error: Annotations file not found at {annotations_path}")
        return {"status": "REQUIRES_DATA"}

    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return {"status": "REQUIRES_DATA"}

    print("Loading model and dataset...")
    model = YOLO(str(model_path))

    with open(annotations_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Normalize category names
    target_classes = sorted(list(CLASS_NAME_ALIASES.keys()))

    # Build ground truth map: list of {id, image_id, class_name, bbox: [x1, y1, x2, y2]}
    all_gts = []
    for ann in coco["annotations"]:
        cat_name = categories.get(ann["category_id"])
        if not cat_name:
            continue
        # Find standardized class name
        std_name = None
        for canonical, aliases in CLASS_NAME_ALIASES.items():
            if cat_name in aliases:
                std_name = canonical
                break
        if not std_name:
            continue

        # COCO bbox: [x, y, width, height] -> [x1, y1, x2, y2]
        x, y, w, h = ann["bbox"]
        all_gts.append({
            "id": ann["id"],
            "image_id": ann["image_id"],
            "class_name": std_name,
            "bbox": [x, y, x + w, y + h],
        })

    print(f"Total images: {len(images)}")
    print(f"Total ground truth annotations: {len(all_gts)}")

    # Run predictions on all images
    all_predictions = []
    print("\nRunning YOLO detection on subset images...")
    for img_id, img_info in images.items():
        img_file = images_dir / img_info["file_name"]
        if not img_file.exists():
            continue

        # Run inference
        results = model.predict(source=str(img_file), conf=0.25, verbose=False)
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_idx = int(boxes.cls[i].item())
                cls_name = r.names[cls_idx]
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()

                std_name = None
                for canonical, aliases in CLASS_NAME_ALIASES.items():
                    if cls_name.lower() in [a.lower() for a in aliases]:
                        std_name = canonical
                        break

                if std_name in target_classes:
                    all_predictions.append({
                        "image_id": img_id,
                        "class_name": std_name,
                        "confidence": conf,
                        "bbox": xyxy,
                    })

    print(f"Total detections generated: {len(all_predictions)}")

    # Calculate per-class metrics
    per_class_results = {}
    all_matched_ious = []
    all_ap50 = []
    all_ap50_95 = []

    iou_thresholds = np.arange(0.50, 0.96, 0.05)

    for cname in target_classes:
        c_gts = [gt for gt in all_gts if gt["class_name"] == cname]
        c_preds = [pred for pred in all_predictions if pred["class_name"] == cname]

        if len(c_gts) == 0:
            continue

        # AP@50
        prec50, rec50, ap50, ious50 = compute_ap_for_class(c_preds, c_gts, iou_threshold=0.50)
        all_matched_ious.extend(ious50)
        all_ap50.append(ap50)

        # AP@50:95 (average over 10 IoU thresholds)
        ap_list = []
        for iou_thresh in iou_thresholds:
            _, _, ap_thresh, _ = compute_ap_for_class(c_preds, c_gts, iou_threshold=iou_thresh)
            ap_list.append(ap_thresh)
        ap50_95 = float(np.mean(ap_list)) if ap_list else 0.0
        all_ap50_95.append(ap50_95)

        mean_iou_class = float(np.mean(ious50)) if ious50 else 0.0

        per_class_results[cname] = {
            "gt_count": len(c_gts),
            "pred_count": len(c_preds),
            "precision": prec50,
            "recall": rec50,
            "mean_iou": mean_iou_class,
            "ap50": ap50,
            "ap50_95": ap50_95,
        }

    overall_map50 = float(np.mean(all_ap50)) if all_ap50 else 0.0
    overall_map50_95 = float(np.mean(all_ap50_95)) if all_ap50_95 else 0.0
    overall_mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0
    overall_prec = float(np.mean([r["precision"] for r in per_class_results.values()])) if per_class_results else 0.0
    overall_rec = float(np.mean([r["recall"] for r in per_class_results.values()])) if per_class_results else 0.0

    return {
        "status": "COMPLETED",
        "num_images": len(images),
        "total_gts": len(all_gts),
        "total_preds": len(all_predictions),
        "overall": {
            "precision": overall_prec,
            "recall": overall_rec,
            "mean_iou": overall_mean_iou,
            "mAP50": overall_map50,
            "mAP50_95": overall_map50_95,
        },
        "per_class": per_class_results,
    }


def print_evaluation_report(results: Dict):
    """Print formatted markdown table for the report."""
    print("\n" + "=" * 80)
    print("OBJECT DETECTION BENCHMARK EVALUATION (COCO 2017 VALIDATION SUBSET)")
    print("=" * 80)
    print(f"Dataset Subset Size   : {results['num_images']} images")
    print(f"Ground Truth Objects  : {results['total_gts']} boxes")
    print(f"Total Detections      : {results['total_preds']} boxes")
    print("-" * 80)
    print(f"{'Class Name':<15} | {'GT':<5} | {'Pred':<5} | {'Precision':<10} | {'Recall':<10} | {'Mean IoU':<10} | {'mAP@50':<10} | {'mAP@50:95':<10}")
    print("-" * 80)

    for cname, metrics in results["per_class"].items():
        print(
            f"{cname:<15} | {metrics['gt_count']:<5} | {metrics['pred_count']:<5} | "
            f"{metrics['precision']:<10.3f} | {metrics['recall']:<10.3f} | "
            f"{metrics['mean_iou']:<10.3f} | {metrics['ap50']:<10.3f} | {metrics['ap50_95']:<10.3f}"
        )

    print("-" * 80)
    ov = results["overall"]
    print(
        f"{'OVERALL (Mean)':<15} | {results['total_gts']:<5} | {results['total_preds']:<5} | "
        f"{ov['precision']:<10.3f} | {ov['recall']:<10.3f} | "
        f"{ov['mean_iou']:<10.3f} | {ov['mAP50']:<10.3f} | {ov['mAP50_95']:<10.3f}"
    )
    print("=" * 80 + "\n")


def main():
    results = evaluate_vision_dataset(ANNOTATIONS_PATH, IMAGES_DIR, MODEL_PATH)
    if results.get("status") == "COMPLETED":
        print_evaluation_report(results)
    else:
        print(f"Evaluation status: {results.get('status')}")


if __name__ == "__main__":
    main()
