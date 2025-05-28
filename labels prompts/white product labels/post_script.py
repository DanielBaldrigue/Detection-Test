import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score

def convert_rel_to_abs(x, y, w, h, img_w, img_h):
    x1 = x / 100 * img_w
    y1 = y / 100 * img_h
    x2 = (x + w) / 100 * img_w
    y2 = (y + h) / 100 * img_h
    return [x1, y1, x2, y2]

def is_box_inside(inner, outer):
    """Return True if `inner` box is fully inside `outer` box."""
    return (
        outer[0] <= inner[0] and
        outer[1] <= inner[1] and
        outer[2] >= inner[2] and
        outer[3] >= inner[3]
    )

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter + 1e-6)


def load_ground_truth(json_path):
    with open(json_path) as f:
        data = json.load(f)

    gt_boxes = defaultdict(list)
    for item in data:
        import re
        # Extract "label_Data_*.png" from full path
        full_path = item.get("image", "")
        match = re.search(r"(label_Data_\d+_\d+\.png)", full_path)
        if not match:
            continue
        img_name = match.group(1)

        if "label" not in item or not item["label"]:
            continue
        annotations = item["label"]
        img_w = annotations[0].get("original_width", 1280)
        img_h = annotations[0].get("original_height", 720)
        for ann in annotations:
            if "rectanglelabels" in ann and "label" in ann["rectanglelabels"]:
                labels = ann["rectanglelabels"]
                if "label" in labels:
                    box = convert_rel_to_abs(ann["x"], ann["y"], ann["width"], ann["height"], img_w, img_h)
                    gt_boxes[img_name].append({"bbox": box, "matched": False})
    return gt_boxes

def load_predictions(pred_dir):
    pred_boxes = defaultdict(list)
    for pred_file in glob.glob(os.path.join(pred_dir, "*.json")):
        with open(pred_file) as f:
            data = json.load(f)
        img_name = os.path.basename(data["image"]).strip()
        for box in data["bboxes"]:
            if box["label"] == "white product labels":
                pred_boxes[img_name].append({
                    "bbox": box["bbox"],
                    "score": box["text_score"]
                })
    return pred_boxes

def filter_large_enclosing_boxes(pred_boxes_dict):
    """
    Removes large predicted boxes that completely enclose another one.
    Keeps the smaller, inner box in such cases.
    Returns:
        - filtered_dict: final filtered predictions
        - removed_dict: dict of removed predictions for inspection
    """
    filtered_dict = {}
    removed_dict = {}

    for img_name, boxes in pred_boxes_dict.items():
        keep = [True] * len(boxes)
        removed = []

        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i != j and keep[j]:  # j may enclose i
                    if is_box_inside(boxes[i]["bbox"], boxes[j]["bbox"]):
                        keep[j] = False
                        removed.append(boxes[j])  # store the removed outer box

        filtered_boxes = [box for box, k in zip(boxes, keep) if k]
        filtered_dict[img_name] = filtered_boxes

        if removed:
            removed_dict[img_name] = removed

    return filtered_dict, removed_dict

def match_predictions(gt_boxes, pred_boxes, iou_thresh=0.3):  # Lowered IoU threshold
    y_true, y_scores, matched_ious = [], [], []

    for img, preds in pred_boxes.items():
        gts = gt_boxes.get(img, [])
        for pred in sorted(preds, key=lambda x: -x["score"]):
            best_iou, best_gt = 0, None
            for gt in gts:
                if gt["matched"]:
                    continue
                iou_val = iou(pred["bbox"], gt["bbox"])
                if iou_val > best_iou:
                    best_iou, best_gt = iou_val, gt
            if best_iou >= iou_thresh:
                best_gt["matched"] = True
                y_true.append(1)
                matched_ious.append(best_iou)
            else:
                y_true.append(0)
                matched_ious.append(0)
            y_scores.append(pred["score"])
    return np.array(y_true), np.array(y_scores), np.array(matched_ious)


# Load previously saved filtered TPs
with open("filtered_true_positives.json") as f:
    filtered_tps = json.load(f)

# Reload GT and predictions (post-filtering)
gt = load_ground_truth("project-2-at-2025-05-11-20-44-75ac7d14.json")
preds = load_predictions("white_product_labels_annotation")
preds, _ = filter_large_enclosing_boxes(preds)
y_true, y_scores, _ = match_predictions(gt, preds)

# Count final GT matches (for comparison)
matched_gt_boxes = {
    img_name: [gt["bbox"] for gt in boxes if gt.get("matched")] 
    for img_name, boxes in gt.items()
}

# Verify which filtered TP boxes would still count as valid TPs
real_filtered_tps = {}

for img, boxes in filtered_tps.items():
    gt_boxes = gt.get(img, [])
    for pred in boxes:
        pred_box = pred["bbox"]
        for gt_entry in gt_boxes:
            if not gt_entry.get("matched") and iou(pred_box, gt_entry["bbox"]) >= 0.3:
                real_filtered_tps.setdefault(img, []).append(pred)
                gt_entry["matched"] = True
                break  # stop after matching one GT

# Save only real filtered TPs
with open("filtered_true_positives_cleaned.json", "w") as f:
    json.dump(real_filtered_tps, f, indent=2)

print(f"Verified TP count among filtered: {sum(len(v) for v in real_filtered_tps.values())}")
