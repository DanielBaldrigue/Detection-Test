import os
import json
import glob

# Load ground truth
with open("project-2-at-2025-05-11-20-44-75ac7d14.json", "r") as f:
    ground_truth_data = json.load(f)

# Paths
preds_dir = "rectangular_cardboard_carton_bboxes_output"
filtered_preds_dir = "filtered_predictions"

# Helper: get filename from "image" field
def extract_image_name(image_path):
    return os.path.splitext(os.path.basename(image_path))[0] + ".json"

# Helper: convert relative to absolute
def to_absolute(b, width, height):
    return [
        b["x"] / 100 * width,
        b["y"] / 100 * height,
        (b["x"] + b["width"]) / 100 * width,
        (b["y"] + b["height"]) / 100 * height
    ]

# IoU calculation
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# Match predictions
for gt in ground_truth_data:
    if "label" not in gt or not gt["label"]:
        continue  # Skip if there are no ground-truth labels

    image_name = extract_image_name(gt["image"])
    orig_w = gt["label"][0]["original_width"]
    orig_h = gt["label"][0]["original_height"]

    # Get GT boxes for 'label'
    gt_boxes = [
        to_absolute(l, orig_w, orig_h)
        for l in gt["label"]
        if l["rectanglelabels"][0] == "label"
    ]

    pred_file = os.path.join(preds_dir, image_name.replace(".png", ".json"))
    filtered_file = os.path.join(filtered_preds_dir, image_name.replace(".png", ".json"))

    if not os.path.exists(pred_file) or not os.path.exists(filtered_file):
        continue

    with open(pred_file, "r") as f:
        preds = json.load(f)
    with open(filtered_file, "r") as f:
        filtered_preds = json.load(f)

    def to_box(p):
        return [p["bbox"][0], p["bbox"][1], p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]]

    # True Positives in original predictions
    tp_original = []
    for p in preds:
        for gt_box in gt_boxes:
            if iou(to_box(p), gt_box) >= 0.5:
                tp_original.append(p)
                break

    # True Positives in filtered predictions
    tp_filtered = []
    for p in filtered_preds:
        for gt_box in gt_boxes:
            if iou(to_box(p), gt_box) >= 0.5:
                tp_filtered.append(p)
                break

    # Find TP removed by filter
    removed = [p for p in tp_original if p not in tp_filtered]

    if removed:
        print(f"\nImage: {image_name}")
        print(f"→ Removed true positives (IoU >= 0.5):")
        for p in removed:
            print(json.dumps(p, indent=2))
