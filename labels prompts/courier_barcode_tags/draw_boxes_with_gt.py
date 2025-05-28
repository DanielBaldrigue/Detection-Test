import os
import cv2
import json
import glob
from collections import defaultdict

def convert_rel_to_abs(x, y, w, h, img_w, img_h):
    x1 = int(x / 100 * img_w)
    y1 = int(y / 100 * img_h)
    x2 = int((x + w) / 100 * img_w)
    y2 = int((y + h) / 100 * img_h)
    return [x1, y1, x2, y2]

def load_ground_truth(gt_file):
    with open(gt_file) as f:
        data = json.load(f)

    gt_boxes = defaultdict(list)
    for item in data:
        import re
        match = re.search(r"(label_Data_\d+_\d+\.png)", item.get("image", ""))
        if not match:
            continue
        img_name = match.group(1)

        annotations = item.get("label", [])
        if not annotations:
            continue
        img_w = annotations[0].get("original_width", 1280)
        img_h = annotations[0].get("original_height", 720)
        for ann in annotations:
            if "rectanglelabels" in ann and "label" in ann["rectanglelabels"]:
                box = convert_rel_to_abs(
                    ann["x"], ann["y"], ann["width"], ann["height"], img_w, img_h
                )
                gt_boxes[img_name].append(box)
    return gt_boxes

def load_predictions(pred_dir):
    pred_boxes = {}
    for fpath in glob.glob(os.path.join(pred_dir, "*.json")):
        with open(fpath) as f:
            data = json.load(f)
        img_name = os.path.basename(data["image"]).strip()
        boxes = []
        for box in data["bboxes"]:
            if box["label"] == "white product labels":
                boxes.append({
                    "bbox": box["bbox"],
                    "score": box["text_score"]
                })
        pred_boxes[img_name] = boxes
    return pred_boxes

def draw_boxes(image_path, preds, gts, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Image not found: {image_path}")
        return

    # Draw ground-truth boxes in green
    for box in gts:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw predicted boxes in blue
    for pred in preds:
        x1, y1, x2, y2 = map(int, pred["bbox"])
        score = pred["score"]
        label = f"{score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

def visualize_predictions_with_gt(pred_dir, gt_file, image_dir, output_dir):
    preds = load_predictions(pred_dir)
    gt_boxes = load_ground_truth(gt_file)

    for img_name, pred_boxes in preds.items():
        image_filename = img_name.replace("label_", "box_")
        # This assumes img_name is like "label_Data_20250509_154006.png"
        image_path = os.path.join(image_dir, img_name)
        output_path = os.path.join(output_dir, image_filename)
        gt = gt_boxes.get(img_name, [])
        draw_boxes(image_path, pred_boxes, gt, output_path)
    print(f"Found {len(preds)} prediction files")
    print(f"Found {len(gt_boxes)} images with ground-truth boxes")


# === USAGE ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Directory of prediction JSONs")
    parser.add_argument("--gt_file", required=True, help="Ground-truth JSON file")
    parser.add_argument("--image_dir", required=True, help="Directory of original images")
    parser.add_argument("--output_dir", required=True, help="Directory to save output images")
    args = parser.parse_args()

    visualize_predictions_with_gt(args.pred_dir, args.gt_file, args.image_dir, args.output_dir)
