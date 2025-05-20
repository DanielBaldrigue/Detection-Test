import os
import json
import cv2
import matplotlib.pyplot as plt

# Customize these paths
image_dir = "Images/"
prediction_file_dir = "white_product_labels_annotation"
filtered_tp_file = "filtered_true_positives_cleaned.json"
output_dir = "filtered_tp_visuals"
os.makedirs(output_dir, exist_ok=True)

# Load filtered true positives
with open(filtered_tp_file) as f:
    filtered_tps = json.load(f)

# Plotting
for img_name, boxes in filtered_tps.items():
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load: {img_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load all predictions for this image for context
    pred_path = os.path.join(prediction_file_dir, img_name.replace(".png", ".json"))
    other_boxes = []
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            pred_data = json.load(f)
            other_boxes = [box["bbox"] for box in pred_data["bboxes"] if box["label"] == "white product labels"]

    # Plot all other predictions in blue
    for box in other_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 120, 255), 2)  # Orange-blue for others

    # Plot filtered true positives in red
    for box in boxes:
        x1, y1, x2, y2 = map(int, box["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Red box

    # Save the image
    out_path = os.path.join(output_dir, f"vis_{img_name}")
    cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization: {out_path}")
