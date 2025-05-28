import os
import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score

used_prompt = "box"
used_prompt_with_underscore = used_prompt.replace(" ", "_")
min_score = 0.1284

def box_is_inside(boxA, boxB):
    """Check if boxA is fully inside boxB."""
    return (boxA[0] >= boxB[0] and boxA[1] >= boxB[1] and
            boxA[2] <= boxB[2] and boxA[3] <= boxB[3])

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

def filter_predictions_with_multiple_labels(pred_boxes, gt_boxes):
    """
    Removes predicted boxes that enclose more than one ground-truth 'label' (box).
    """
    filtered_preds = defaultdict(list)

    def label_bb_is_inside_box_bounding_box(pred_box, label_box):
        # Check if the ground truth box is fully inside the predicted box
        return (label_box[0] >= pred_box[0] and label_box[1] >= pred_box[1] and
                label_box[2] <= pred_box[2] and label_box[3] <= pred_box[3])

    for img_name, boxes in pred_boxes.items():
        gts = [b["bbox"] for b in gt_boxes.get(img_name, [])]
        for pred in boxes:
            pred_box = pred["bbox"]
            count = sum(1 for gt in gts if label_bb_is_inside_box_bounding_box(pred_box, gt))
            if count <= 1:  # Keep only if at most one GT box is inside
                filtered_preds[img_name].append(pred)
            #else:
                #print(f"Removing pred box {pred_box} in image {img_name} as it encloses multiple GT boxes.")
    return filtered_preds

def delete_large_boxes_bounding_boxes_3_no_node(pred_boxes_dict):
    filtered_pred_boxes = defaultdict(list)

    for img_name, boxes in pred_boxes_dict.items():
        boxes_to_keep = []
        for i in range(len(boxes)):
            keep = True
            for j in range(len(boxes)):
                if i != j:
                    if box_is_inside(boxes[i]["bbox"], boxes[j]["bbox"]):
                        keep = False
                        break
            if keep:
                boxes_to_keep.append(boxes[i])
        filtered_pred_boxes[img_name] = boxes_to_keep

    return filtered_pred_boxes


def delete_remaining_lonely_boxes(pred_boxes_dict, gt_boxes):
    filtered_pred_boxes = defaultdict(list)

    for img_name, pred_boxes in pred_boxes_dict.items():
        gt_boxes_list = gt_boxes.get(img_name, [])
        boxes_to_keep = []

        for pred_box in pred_boxes:
            keep = False
            for gt_box in gt_boxes_list:
                if box_is_inside(gt_box["bbox"], pred_box["bbox"]):
                    keep = True
                    break
            if keep:
                boxes_to_keep.append(pred_box)

        filtered_pred_boxes[img_name] = boxes_to_keep

    return filtered_pred_boxes

def compute_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def delete_highest_area_boxes_for_same_label(pred_boxes_dict, gt_boxes_dict):
    """
    Deletes predicted boxes with higher area if multiple predictions map to the same GT label.
    Assumes pred_boxes only have 'bbox' and 'score', no 'area' or 'mask'.
    """
    filtered_pred_boxes = defaultdict(list)

    for img_name, pred_boxes in pred_boxes_dict.items():
        gt_boxes = gt_boxes_dict.get(img_name, [])
        pairs_label_box = []

        for gt_idx, gt in enumerate(gt_boxes):
            for pred_idx, pred in enumerate(pred_boxes):
                if box_is_inside(gt["bbox"], pred["bbox"]):
                    pairs_label_box.append((gt_idx, pred_idx))

        # Find boxes to delete
        delete_indices = set()
        for (gt_idx1, pred_idx1) in pairs_label_box:
            for (gt_idx2, pred_idx2) in pairs_label_box:
                if gt_idx1 == gt_idx2 and pred_idx1 != pred_idx2:
                    area1 = compute_bbox_area(pred_boxes[pred_idx1]["bbox"])
                    area2 = compute_bbox_area(pred_boxes[pred_idx2]["bbox"])
                    if area1 > area2:
                        delete_indices.add(pred_idx1)

        # Filter out deleted boxes
        filtered = [box for i, box in enumerate(pred_boxes) if i not in delete_indices]
        filtered_pred_boxes[img_name] = filtered

    return filtered_pred_boxes

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
            if "rectanglelabels" in ann and "box" in ann["rectanglelabels"]:
                boxes = ann["rectanglelabels"]
                if "box" in boxes:
                    box = convert_rel_to_abs(ann["x"], ann["y"], ann["width"], ann["height"], img_w, img_h)
                    gt_boxes[img_name].append({"bbox": box, "matched": False})
    return gt_boxes

def load_ground_truth_labels(json_path):
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
            if box["label"] == used_prompt and box["text_score"] >= min_score:
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

def plot_pr_curve(y_true, y_scores, out_dir="plots_with_filter"):
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

def plot_f1_score(y_true, y_scores, out_dir="plots_with_filter"):
    if len(np.unique(y_true)) < 2:
        print("⚠️ Cannot plot F1 curve: not enough positive labels.")
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

def plot_score_histograms(y_true, y_scores, out_dir="plots_with_filter"):
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

def plot_calibration(y_true, y_scores, bins=10, out_dir="plots_with_filter"):
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

def save_predictions_to_folder(pred_boxes_dict, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for image_name, predictions in pred_boxes_dict.items():
        # Remove file extension and use it as filename
        base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(output_folder, base_name + ".json")

        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

    print(f"Saved predictions for {len(pred_boxes_dict)} images to '{output_folder}'")

def draw_boxes_on_images(image_folder, output_folder, pred_boxes_dict, gt_boxes):
    os.makedirs(output_folder, exist_ok=True)

    for image_name in pred_boxes_dict:
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not load image: {image_path}")
            continue

        # Draw ground truth boxes (green)
        for gt in gt_boxes.get(image_name, []):
            x1, y1, x2, y2 = map(int, gt["bbox"])
            label = gt.get("label", "GT")  # use default if missing
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw predicted boxes (red)
        for pred in pred_boxes_dict.get(image_name, []):
            x1, y1, x2, y2 = map(int, pred["bbox"])
            label = pred.get("label", "Pred")
            score = pred.get("score", 0.0)
            label_text = f'{label} ({score:.2f})'
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, label_text, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Save the annotated image
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, image)

    print(f"Saved annotated images to '{output_folder}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", default="project-2-at-2025-05-11-20-44-75ac7d14.json", help="Path to ground-truth JSON file")
    parser.add_argument("--pred_dir", default= used_prompt_with_underscore + "_bboxes_output", help="Directory containing prediction JSONs")
    args = parser.parse_args()

    gt = load_ground_truth(args.gt_file)
    preds = load_predictions(args.pred_dir)
    labels = load_ground_truth_labels(args.gt_file)
    #preds, filtered_tp_preds = filter_large_enclosing_boxes_and_track_tps(preds, gt, iou_thresh=0.3)
    filtered_preds_1 = filter_predictions_with_multiple_labels(preds, labels)

    #filtered_preds_2 = delete_large_boxes_bounding_boxes_3_no_node(filtered_preds_1)

    filtered_preds_3 = delete_remaining_lonely_boxes(filtered_preds_1, labels)

    filtered_preds_4 = delete_highest_area_boxes_for_same_label(filtered_preds_3, labels)

    save_predictions_to_folder(filtered_preds_3, "filtered_predictions")
    draw_boxes_on_images(image_folder="Images", output_folder="Results",
    pred_boxes_dict=filtered_preds_4,
    gt_boxes=gt
)


    # Save removed true positive predictions
    #with open("filtered_true_positives.json", "w") as f:
    #    json.dump(filtered_tp_preds, f, indent=2)

    y_true, y_scores, ious = match_predictions(gt, filtered_preds_4)

    # Step 1: Find the minimum score among TPs
    tp_scores = y_scores[y_true == 1]

    if len(tp_scores) == 0:
        print("⚠️ No true positives found. Skipping threshold filtering.")
    else:
        min_tp_score = np.min(tp_scores)

    # Step 2: Count FPs with score >= min_tp_score
    fp_scores = y_scores[y_true == 0]
    high_score_fp_count = np.sum(fp_scores >= min_tp_score)

    print(f"✅ Least TP score: {min_tp_score:.4f}")
    print(f"📦 False Positives with score ≥ {min_tp_score:.4f}: {high_score_fp_count}")

    # Step 3: Filter predictions with score ≥ min_tp_score
    filtered_preds = defaultdict(list)
    for img_name, boxes in filtered_preds_3.items():
        for box in boxes:
            if box["score"] >= min_tp_score:
                filtered_preds[img_name].append(box)

    # Step 4: Save filtered predictions to JSON
    output_dir = "filtered_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name, boxes in filtered_preds.items():
        out_path = os.path.join(output_dir, img_name.replace(".png", ".json"))
        with open(out_path, "w") as f:
            json.dump({
                "image": img_name,
                "bboxes": [
                    {
                        "bbox": box["bbox"],
                        "text_score": box["score"],
                        "label": used_prompt
                    }
                    for box in boxes
                ]
            }, f, indent=2)

    print(f"✅ Filtered predictions saved to '{output_dir}/'")
    
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




