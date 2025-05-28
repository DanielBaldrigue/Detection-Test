import os
import json
import cv2

# === CONFIG ===
image_dir = "Images"  # Directory containing images
prediction_dir = "filtered_predictions"
ground_truth_file = "project-2-at-2025-05-11-20-44-75ac7d14.json"
output_dir = "output_images"  # Directory to save annotated images
os.makedirs(output_dir, exist_ok=True)

# === LOAD AND INDEX GT DATA ===
with open(ground_truth_file, "r") as f:
    gt_raw = json.load(f)

# Index GT by image filename (e.g., extract "label_Data_20250509_135650.png")
gt_data = {}
for item in gt_raw:
    img_path = item["image"].replace("\\", "/")
    filename_tail = os.path.basename(img_path)  # e.g., label_Data_20250507_131603.png
    if "label" in item:
        gt_data[filename_tail] = item["label"]
    else:
        print(f"⚠️  Skipping {filename_tail} — no 'label' field")

# === DRAW HELPER ===
def draw_box(img, bbox, label, color, thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# === PROCESS IMAGES ===
for image_name in os.listdir(image_dir):
    if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(image_dir, image_name)
    pred_path = os.path.join(prediction_dir, os.path.splitext(image_name)[0].replace("box_", "") + ".json")

    img = cv2.imread(img_path)
    if img is None:
        print(f"Couldn't read {img_path}")
        continue

    height, width = img.shape[:2]

    # Try to match using filename tail
    matching_gt_key = None
    for key in gt_data:
        if image_name.endswith(key):
            matching_gt_key = key
            break

    # Draw GT (green)
    if matching_gt_key:
        for ann in gt_data.get(image_name, []):
            x_pct = ann["x"] / 100.0
            y_pct = ann["y"] / 100.0
            w_pct = ann["width"] / 100.0
            h_pct = ann["height"] / 100.0

            x1 = x_pct * width
            y1 = y_pct * height
            x2 = x1 + w_pct * width
            y2 = y1 + h_pct * height

            label = ann["rectanglelabels"][0]
            draw_box(img, (x1, y1, x2, y2), f"GT: {label}", (0, 255, 0))
    """
    # Draw Predictions (red)
    if os.path.exists(pred_path):
        with open(pred_path, "r") as f:
            preds = json.load(f)

        for pred in preds.get("bboxes", []):
            bbox = pred["bbox"]
            label = pred.get("label", "pred")
            score = pred.get("text_score", 0)
            draw_box(img, bbox, f"PR: {label} ({score:.2f})", (0, 0, 255))
    else:
        print(f"No prediction for {image_name}")
    """
    # Save
    out_path = os.path.join(output_dir, image_name)
    cv2.imwrite(out_path, img)

print("✅ Done drawing boxes.")
