import os
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

# Standard Pascal VOC 20 classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Mapping to reconcile label naming differences (e.g. COCO vs Pascal VOC)
VOC_NAME_MAP = {
    "potted plant": "pottedplant",
    "dining table": "diningtable",
    "tv monitor": "tvmonitor",
    "tv": "tvmonitor",
    "aeroplane": "aeroplane",
    "airplane": "aeroplane",
    "motorbike": "motorbike",
    "motorcycle": "motorbike",
    "sofa": "sofa",
    "couch": "sofa"
}

def normalize_label(name: str) -> str:
    """Normalizes class names by stripping spaces, hyphens, and underscores."""
    cleaned = str(name).strip().lower()
    if cleaned in VOC_NAME_MAP:
        return VOC_NAME_MAP[cleaned]
    return cleaned.replace(" ", "").replace("_", "").replace("-", "")

def calculate_iou(boxA, boxB):
    """Computes IoU between two boxes [xmin, ymin, xmax, ymax]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union_area = areaA + areaB - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def parse_voc_xml(xml_path):
    """Extracts ground truth bounding boxes and class labels from a Pascal VOC XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    width = int(size_elem.find("width").text) if size_elem is not None else 1
    height = int(size_elem.find("height").text) if size_elem is not None else 1

    objects = []
    for obj in root.findall("object"):
        raw_name = obj.find("name").text
        cls_name = normalize_label(raw_name)

        if cls_name not in VOC_CLASSES:
            continue

        difficult_elem = obj.find("difficult")
        is_difficult = int(difficult_elem.text) if difficult_elem is not None else 0

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        objects.append({
            "name": cls_name,
            "bbox": [xmin, ymin, xmax, ymax],
            "difficult": is_difficult
        })
    return objects, width, height

def classify_object_size(box_area, total_img_area):
    """Categorizes target scale into Small, Medium, or Large."""
    ratio = box_area / max(1.0, total_img_area)
    if ratio < 0.04:
        return "Small Scale (<4%)"
    elif ratio < 0.20:
        return "Medium Scale (4-20%)"
    return "Large Scale (>20%)"

def main():
    base_dir = Path(__file__).resolve().parent
    voc_devkit = base_dir / "VOC_dataset" / "VOCdevkit"
    dataset_splits = [voc_devkit / "VOC2007", voc_devkit / "VOC2012"]

    model_weight = base_dir / "yolo26n.pt"
    if not model_weight.exists():
        model_weight = base_dir / "yolov8n.pt"

    print(f"Loading weights from: {model_weight}")
    model = YOLO(str(model_weight))

    records = []
    max_images_per_split = 250  # Adjust as needed

    for split in dataset_splits:
        ann_dir = split / "Annotations"
        img_dir = split / "JPEGImages"

        if not ann_dir.exists() or not img_dir.exists():
            print(f"Directory not found, skipping: {split}")
            continue

        split_name = split.name
        xml_files = sorted(list(ann_dir.glob("*.xml")))[:max_images_per_split]
        print(f"Evaluating {len(xml_files)} samples from {split_name}...")

        for idx, xml_path in enumerate(xml_files):
            sample_id = f"{split_name}_{xml_path.stem}"
            img_path = img_dir / f"{xml_path.stem}.jpg"
            if not img_path.exists():
                continue

            gt_objects, img_w, img_h = parse_voc_xml(xml_path)
            total_area = float(img_w * img_h)

            results = model.predict(str(img_path), conf=0.25, iou=0.45, verbose=False)
            res = results[0]

            pred_boxes = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls_id = int(b.cls[0].item())
                    raw_pred_name = res.names.get(cls_id, str(cls_id))
                    norm_pred_name = normalize_label(raw_pred_name)

                    pred_boxes.append({
                        "class": norm_pred_name,
                        "bbox": b.xyxy[0].tolist(),
                        "conf": float(b.conf[0].item()),
                        "matched": False
                    })

            # Match Ground Truth items against Predictions
            for gt in gt_objects:
                if gt["difficult"] == 1:
                    continue

                gt_box = gt["bbox"]
                box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                condition = classify_object_size(box_area, total_area)
                gt_cls = gt["name"]

                best_iou = 0.0
                best_idx = -1
                for p_idx, pb in enumerate(pred_boxes):
                    if pb["matched"]:
                        continue
                    iou = calculate_iou(gt_box, pb["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = p_idx

                if best_iou >= 0.5 and best_idx != -1:
                    pred = pred_boxes[best_idx]
                    pred["matched"] = True
                    outcome = "TP" if pred["class"] == gt_cls else "Misclassification"

                    records.append({
                        "Sample_ID": sample_id,
                        "Split": split_name,
                        "Ground_Truth": gt_cls,
                        "Predicted_Class": pred["class"],
                        "Outcome": outcome,
                        "Confidence": round(pred["conf"], 3),
                        "IoU": round(best_iou, 3),
                        "Environmental_Condition": condition
                    })
                else:
                    records.append({
                        "Sample_ID": sample_id,
                        "Split": split_name,
                        "Ground_Truth": gt_cls,
                        "Predicted_Class": "None",
                        "Outcome": "Missed Target",
                        "Confidence": 0.0,
                        "IoU": round(best_iou, 3),
                        "Environmental_Condition": condition
                    })

            # Collect unmatched detections as False Positives
            for pb in pred_boxes:
                if not pb["matched"] and pb["class"] in VOC_CLASSES:
                    p_area = (pb["bbox"][2] - pb["bbox"][0]) * (pb["bbox"][3] - pb["bbox"][1])
                    condition = classify_object_size(p_area, total_area)
                    records.append({
                        "Sample_ID": sample_id,
                        "Split": split_name,
                        "Ground_Truth": "None (Background)",
                        "Predicted_Class": pb["class"],
                        "Outcome": "False Positive",
                        "Confidence": round(pb["conf"], 3),
                        "IoU": 0.0,
                        "Environmental_Condition": condition
                    })

    out_file = base_dir / "voc_behavioral_results.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(str(out_file), index=False, encoding="utf-8")
    print(f"\nDone! Processed {len(df)} records. Saved to: {out_file}")

if __name__ == "__main__":
    main()