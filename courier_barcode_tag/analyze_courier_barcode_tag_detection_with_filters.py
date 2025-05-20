import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score

def is_box_inside(inner, outer):
    """Return True if `inner` box is fully inside `outer` box."""
    return (
        outer[0] <= inner[0] and
        outer[1] <= inner[1] and
        outer[2] >= inner[2] and
        outer[3] >= inner[3]
    )

def filter_large_enclosing_boxes_and_track_tps(pred_boxes_dict, gt_boxes_dict, iou_thresh=0.5):
    filtered_dict = {}
    filtered_true_positives = {}

    for img_name, boxes in pred_boxes_dict.items():
        boxes = sorted(boxes, key=lambda b: -b["score"])
        keep = [True] * len(boxes)
        is_tp = [False] * len(boxes)
        removed_tp = []

        import copy
        gt_boxes = copy.deepcopy(gt_boxes_dict.get(img_name, []))
        for i, pred in enumerate(boxes):
            for gt in gt_boxes:
                if not gt.get("matched") and iou(pred["bbox"], gt["bbox"]) >= iou_thresh:
                    is_tp[i] = True
                    gt["matched"] = True
                    break

        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i != j and keep[j]:
                    # Only remove the outer box if it's lower confidence than the inner one
                    if is_box_inside(boxes[i]["bbox"], boxes[j]["bbox"]) and boxes[i]["score"] > boxes[j]["score"]:
                        keep[j] = False
                        if is_tp[j]:
                            removed_tp.append(boxes[j])

        filtered_boxes = [box for box, k in zip(boxes, keep) if k]
        filtered_dict[img_name] = filtered_boxes
        if removed_tp:
            filtered_true_positives[img_name] = removed_tp

    return filtered_dict, filtered_true_positives

def convert_rel_to_abs(x, y, w, h, img_w, img_h):
    x1 = x / 100 * img_w
    y1 = y / 100 * img_h
    x2 = (x + w) / 100 * img_w
    y2 = (y + h) / 100 * img_h
    return [x1, y1, x2, y2]

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
            if box["label"] == "courier barcode tag":
                pred_boxes[img_name].append({
                    "bbox": box["bbox"],
                    "score": box["text_score"]
                })
    return pred_boxes

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

def plot_pr_curve(y_true, y_scores):
    if len(np.unique(y_true)) < 2:
        print("⚠️ Cannot plot PR curve: no positive matches found.")
        return
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_f1_score(y_true, y_scores):
    if len(np.unique(y_true)) < 2:
        print("⚠️ Cannot plot F1 curve: not enough positive labels.")
        return
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    plt.figure()
    plt.plot(thresholds, f1[:-1])
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Threshold")
    plt.grid(True)
    plt.show()

def plot_score_histograms(y_true, y_scores):
    if len(y_scores) == 0:
        print("⚠️ No predictions to plot.")
        return
    plt.figure()
    plt.hist(y_scores[y_true == 1], bins=20, alpha=0.6, label="True Positives")
    plt.hist(y_scores[y_true == 0], bins=20, alpha=0.6, label="False Positives")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_calibration(y_true, y_scores, bins=10):
    if len(y_scores) == 0:
        return
    bin_edges = np.linspace(0, 1, bins + 1)
    accuracies, confidences = [], []
    for i in range(bins):
        mask = (y_scores >= bin_edges[i]) & (y_scores < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_scores[mask].mean()
        accuracies.append(acc)
        confidences.append(conf)
    plt.figure()
    plt.plot(confidences, accuracies, marker='o', label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_iou_distribution(ious, y_true):
    if np.sum(y_true == 1) == 0:
        print("⚠️ No true positives to plot IoU distribution.")
        return
    plt.figure()
    plt.hist(ious[y_true == 1], bins=20)
    plt.xlabel("IoU")
    plt.ylabel("True Positive Count")
    plt.title("IoU Distribution for TPs")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", required=True, help="Path to ground-truth JSON file")
    parser.add_argument("--pred_dir", required=True, help="Directory containing prediction JSONs")
    args = parser.parse_args()

    gt = load_ground_truth(args.gt_file)
    preds = load_predictions(args.pred_dir)
    preds, filtered_tp_preds = filter_large_enclosing_boxes_and_track_tps(preds, gt, iou_thresh=0.3)

    # Save removed true positive predictions
    with open("filtered_true_positives.json", "w") as f:
        json.dump(filtered_tp_preds, f, indent=2)

    y_true, y_scores, ious = match_predictions(gt, preds)

    print("\nSample predicted boxes vs. GT for first image with predictions:")
    for img_name in preds:
        if img_name in gt:
            print(f"\nImage: {img_name}")
            for i, pred in enumerate(preds[img_name][:3]):
                print(f"  Pred #{i+1}: {pred['bbox']} (score: {pred['score']:.2f})")
            for i, gt_box in enumerate(gt[img_name][:3]):
                print(f"  GT   #{i+1}: {gt_box['bbox']}")
            break

    print("\n===== DEBUG REPORT =====")
    print("Ground truth images:", len(gt))
    print("Prediction images:", len(preds))
    print("Total predictions:", len(y_scores))
    print("Matched TPs:", int(np.sum(y_true == 1)))
    print("Unmatched FPs:", int(np.sum(y_true == 0)))
    print("========================\n")

    plot_pr_curve(y_true, y_scores)
    plot_f1_score(y_true, y_scores)
    plot_score_histograms(y_true, y_scores)
    plot_calibration(y_true, y_scores)
    plot_iou_distribution(ious, y_true)


