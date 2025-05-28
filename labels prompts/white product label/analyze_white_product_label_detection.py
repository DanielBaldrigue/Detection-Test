import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score

used_prompt = "white product label"  # Change this to the prompt used for predictions
used_prompt_with_underscore = used_prompt.replace(" ", "_")

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
            if box["label"] == used_prompt:
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

def plot_pr_curve(y_true, y_scores, out_dir="plots_without_filter"):
    if len(np.unique(y_true)) < 2:
        print("⚠️ Cannot plot PR curve: no positive matches found.")
        return
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_f1_score(y_true, y_scores, out_dir="plots_without_filter"):
    if len(np.unique(y_true)) < 2:
        print("⚠️ Cannot plot F1 curve: not enough positive boxes.")
        return
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(thresholds, f1[:-1])
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Threshold")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "f1_score.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_score_histograms(y_true, y_scores, out_dir="plots_without_filter"):
    if len(y_scores) == 0:
        print("⚠️ No predictions to plot.")
        return
    
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.hist(y_scores[y_true == 1], bins=20, alpha=0.6, label="True Positives")
    plt.hist(y_scores[y_true == 0], bins=20, alpha=0.6, label="False Positives")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "score_histograms.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_calibration(y_true, y_scores, bins=10, out_dir="plots_without_filter"):
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

    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(confidences, accuracies, marker='o', label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "calibration.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_iou_distribution(ious, y_true, out_dir="plots_without_filter"):
    if np.sum(y_true == 1) == 0:
        print("⚠️ No true positives to plot IoU distribution.")
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.hist(ious[y_true == 1], bins=20)
    plt.xlabel("IoU")
    plt.ylabel("True Positive Count")
    plt.title("IoU Distribution for TPs")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "iou_distribution.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", default="project-2-at-2025-05-11-20-44-75ac7d14.json", help="Path to ground-truth JSON file")
    parser.add_argument("--pred_dir", default= used_prompt_with_underscore + "_bboxes_output", help="Directory containing prediction JSONs")
    args = parser.parse_args()

    gt = load_ground_truth(args.gt_file)
    preds = load_predictions(args.pred_dir)

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
