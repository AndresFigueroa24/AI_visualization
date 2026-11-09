import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
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
    "couch": "sofa",
}

# Evaluation settings
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MATCH_IOU_THRESHOLD = 0.50
MAX_IMAGES_PER_SPLIT = 250  # Set to None to evaluate all XML files


def normalize_label(name: str) -> str:
    """Normalize class names and reconcile common VOC/COCO naming differences."""
    cleaned = str(name).strip().lower()
    if cleaned in VOC_NAME_MAP:
        return VOC_NAME_MAP[cleaned]
    return cleaned.replace(" ", "").replace("_", "").replace("-", "")


def calculate_iou(box_a, box_b) -> float:
    """Compute IoU for boxes in [xmin, ymin, xmax, ymax] format."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_box_geometry(bbox, img_w: int, img_h: int) -> dict:
    """Compute relative area (%), aspect ratio (w/h), and normalized center coordinates."""
    total_area = float(max(1, img_w * img_h))
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    area = w * h
    rel_area_pct = (area / total_area) * 100.0
    aspect_ratio = w / max(1.0, h)
    cx_norm = ((bbox[0] + bbox[2]) * 0.5) / max(1.0, float(img_w))
    cy_norm = ((bbox[1] + bbox[3]) * 0.5) / max(1.0, float(img_h))
    return {
        "area": area,
        "rel_area_pct": round(rel_area_pct, 3),
        "aspect_ratio": round(aspect_ratio, 3),
        "cx_norm": round(cx_norm, 3),
        "cy_norm": round(cy_norm, 3),
    }


def parse_voc_xml(xml_path: Path):
    """Read Ground Truth boxes, labels and difficult flags from a VOC XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    width = int(size_elem.find("width").text) if size_elem is not None else 1
    height = int(size_elem.find("height").text) if size_elem is not None else 1

    objects = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        bndbox = obj.find("bndbox")
        if name_elem is None or name_elem.text is None or bndbox is None:
            continue

        cls_name = normalize_label(name_elem.text)
        if cls_name not in VOC_CLASSES:
            continue

        difficult_elem = obj.find("difficult")
        is_difficult = int(difficult_elem.text) if difficult_elem is not None and difficult_elem.text else 0

        try:
            bbox = [
                float(bndbox.find("xmin").text),
                float(bndbox.find("ymin").text),
                float(bndbox.find("xmax").text),
                float(bndbox.find("ymax").text),
            ]
        except (AttributeError, TypeError, ValueError):
            continue

        objects.append({"name": cls_name, "bbox": bbox, "difficult": is_difficult})

    return objects, width, height


def classify_object_size(box_area: float, total_img_area: float) -> str:
    """Categorize target scale by its fraction of the full image area."""
    ratio = box_area / max(1.0, total_img_area)
    if ratio < 0.04:
        return "Small Scale (<4%)"
    if ratio < 0.20:
        return "Medium Scale (4-20%)"
    return "Large Scale (>20%)"


def best_prediction_for_ground_truth(gt_box, predictions):
    """Find the highest-IoU prediction among predictions not already matched to a GT."""
    best_iou = 0.0
    best_idx = -1

    for pred_idx, pred in enumerate(predictions):
        if pred["matched"]:
            continue
        overlap = calculate_iou(gt_box, pred["bbox"])
        if overlap > best_iou:
            best_iou = overlap
            best_idx = pred_idx

    if best_idx == -1:
        return None, 0.0
    return predictions[best_idx], best_iou


def main():
    base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

    # Support both layouts:
    #   phase1\VOC_dataset\VOCdevkit\VOC2007
    #   phase1\archive\VOC_dataset\VOCdevkit\VOC2007
    candidate_devkits = [
        base_dir / "VOC_dataset" / "VOCdevkit",
        base_dir / "archive" / "VOC_dataset" / "VOCdevkit",
    ]

    def has_voc_split(devkit: Path) -> bool:
        return any(
            (devkit / split_name / "Annotations").is_dir()
            and (devkit / split_name / "JPEGImages").is_dir()
            for split_name in ("VOC2007", "VOC2012")
        )

    voc_devkit = next((p for p in candidate_devkits if has_voc_split(p)), None)
    if voc_devkit is None:
        checked = "\n".join(f"  - {p}" for p in candidate_devkits)
        raise FileNotFoundError(
            "Could not locate the PASCAL VOC dataset. Checked these locations:\n"
            f"{checked}\n"
            "Each VOC split must contain both 'Annotations' and 'JPEGImages'. "
            "Move the dataset into one of these locations or update candidate_devkits."
        )

    print(f"Using VOC dataset at: {voc_devkit}")
    dataset_splits = [voc_devkit / "VOC2007", voc_devkit / "VOC2012"]

    model_weight = base_dir / "yolo26n.pt"
    if not model_weight.exists():
        model_weight = base_dir / "yolov8n.pt"
    if not model_weight.exists():
        raise FileNotFoundError(
            f"Model weights not found. Expected {base_dir / 'yolo26n.pt'} "
            f"or {base_dir / 'yolov8n.pt'}."
        )

    print(f"Loading weights from: {model_weight}")
    model = YOLO(str(model_weight))

    records = []
    processed_images = 0
    rec_counter = 0

    for split in dataset_splits:
        ann_dir = split / "Annotations"
        img_dir = split / "JPEGImages"

        if not ann_dir.exists() or not img_dir.exists():
            print(f"Directory not found, skipping: {split}")
            continue

        split_name = split.name
        xml_files = sorted(ann_dir.glob("*.xml"))
        if MAX_IMAGES_PER_SPLIT is not None:
            xml_files = xml_files[:MAX_IMAGES_PER_SPLIT]
        print(f"Evaluating {len(xml_files)} samples from {split_name}...")

        for xml_path in xml_files:
            sample_id = f"{split_name}_{xml_path.stem}"
            img_path = img_dir / f"{xml_path.stem}.jpg"
            if not img_path.exists():
                print(f"Image not found, skipping: {img_path}")
                continue

            processed_images += 1
            gt_objects, img_w, img_h = parse_voc_xml(xml_path)
            total_area = float(max(1, img_w * img_h))

            results = model.predict(
                str(img_path),
                conf=CONF_THRESHOLD,
                iou=NMS_IOU_THRESHOLD,
                verbose=False,
            )
            res = results[0]

            predictions = []
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    class_id = int(box.cls[0].item())
                    raw_name = res.names.get(class_id, str(class_id))
                    pred_name = normalize_label(raw_name)
                    predictions.append({
                        "class": pred_name,
                        "bbox": box.xyxy[0].tolist(),
                        "conf": float(box.conf[0].item()),
                        "matched": False,
                        "used_as_low_iou_candidate": False,
                    })

            eligible_gts = [gt for gt in gt_objects if gt["difficult"] != 1]

            # ------------------------------------------------------------------
            # PASS 1: Global IoU-sorted matching for valid matches (IoU >= 0.50)
            # Prevents greedy XML-order from stealing a high-IoU TP for a low-IoU GT
            # ------------------------------------------------------------------
            valid_pairs = []
            for gt_idx, gt in enumerate(eligible_gts):
                for pred_idx, pred in enumerate(predictions):
                    iou_val = calculate_iou(gt["bbox"], pred["bbox"])
                    if iou_val >= MATCH_IOU_THRESHOLD:
                        valid_pairs.append((iou_val, pred["conf"], gt_idx, pred_idx))

            valid_pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)
            matched_gt_map = {}

            for iou_val, _, gt_idx, pred_idx in valid_pairs:
                if gt_idx not in matched_gt_map and not predictions[pred_idx]["matched"]:
                    predictions[pred_idx]["matched"] = True
                    matched_gt_map[gt_idx] = (predictions[pred_idx], iou_val)

            # ------------------------------------------------------------------
            # PASS 2: Record Ground Truth Outcomes (TP, Misclassification, or Missed Target)
            # For unmatched GTs, attach the best remaining low-IoU candidate (0 < IoU < 0.50)
            # ------------------------------------------------------------------
            for gt_idx, gt in enumerate(eligible_gts):
                gt_box = gt["bbox"]
                geom = compute_box_geometry(gt_box, img_w, img_h)
                condition = classify_object_size(geom["area"], total_area)
                gt_class = gt["name"]

                if gt_idx in matched_gt_map:
                    candidate, best_iou = matched_gt_map[gt_idx]
                    outcome = "TP" if candidate["class"] == gt_class else "Misclassification"
                    rec_counter += 1
                    records.append({
                        "Record_ID": f"rec_{rec_counter:05d}",
                        "Sample_ID": sample_id,
                        "Split": split_name,
                        "Ground_Truth": gt_class,
                        "Predicted_Class": candidate["class"],
                        "Candidate_Class": candidate["class"],
                        "Outcome": outcome,
                        "Confidence": round(candidate["conf"], 3),
                        "Candidate_Confidence": round(candidate["conf"], 3),
                        "Confidence_Source": "Observed prediction",
                        "IoU": round(best_iou, 3),
                        "Environmental_Condition": condition,
                        "Rel_Area_Pct": geom["rel_area_pct"],
                        "Aspect_Ratio": geom["aspect_ratio"],
                        "Center_X_Norm": geom["cx_norm"],
                        "Center_Y_Norm": geom["cy_norm"],
                        "FP_Subtype": "Matched GT",
                        "Is_Duplicate_Candidate_Row": False,
                    })
                else:
                    candidate, best_iou = best_prediction_for_ground_truth(gt_box, predictions)
                    has_candidate = candidate is not None and best_iou > 0.0
                    if has_candidate:
                        candidate["used_as_low_iou_candidate"] = True
                    candidate_class = candidate["class"] if has_candidate else "None"
                    candidate_conf = round(candidate["conf"], 3) if has_candidate else np.nan

                    rec_counter += 1
                    records.append({
                        "Record_ID": f"rec_{rec_counter:05d}",
                        "Sample_ID": sample_id,
                        "Split": split_name,
                        "Ground_Truth": gt_class,
                        "Predicted_Class": "None",
                        "Candidate_Class": candidate_class,
                        "Outcome": "Missed Target",
                        "Confidence": candidate_conf,
                        "Candidate_Confidence": candidate_conf,
                        "Confidence_Source": (
                            "Best-overlap candidate below IoU threshold"
                            if has_candidate else "No overlapping candidate"
                        ),
                        "IoU": round(best_iou, 3),
                        "Environmental_Condition": condition,
                        "Rel_Area_Pct": geom["rel_area_pct"],
                        "Aspect_Ratio": geom["aspect_ratio"],
                        "Center_X_Norm": geom["cx_norm"],
                        "Center_Y_Norm": geom["cy_norm"],
                        "FP_Subtype": (
                            "Localization Error (0 < IoU < 0.5)"
                            if has_candidate else "Blind Spot (IoU = 0)"
                        ),
                        "Is_Duplicate_Candidate_Row": False,
                    })

            # ------------------------------------------------------------------
            # PASS 3: Record Unmatched Predictions as False Positives
            # ------------------------------------------------------------------
            eligible_gt_boxes = [gt["bbox"] for gt in eligible_gts]
            for pred in predictions:
                if pred["matched"]:
                    continue
                if pred["class"] not in VOC_CLASSES:
                    continue

                pred_box = pred["bbox"]
                geom = compute_box_geometry(pred_box, img_w, img_h)
                condition = classify_object_size(geom["area"], total_area)
                max_gt_iou = max(
                    (calculate_iou(pred_box, gt_box) for gt_box in eligible_gt_boxes),
                    default=0.0,
                )

                if max_gt_iou == 0.0:
                    fp_subtype = "Pure Background FP (IoU = 0)"
                elif pred["used_as_low_iou_candidate"]:
                    fp_subtype = "Localization Shift FP (0 < IoU < 0.5)"
                else:
                    fp_subtype = "Duplicate / Redundant Box on GT"

                rec_counter += 1
                records.append({
                    "Record_ID": f"rec_{rec_counter:05d}",
                    "Sample_ID": sample_id,
                    "Split": split_name,
                    "Ground_Truth": "None (Background)",
                    "Predicted_Class": pred["class"],
                    "Candidate_Class": pred["class"],
                    "Outcome": "False Positive",
                    "Confidence": round(pred["conf"], 3),
                    "Candidate_Confidence": round(pred["conf"], 3),
                    "Confidence_Source": "Observed prediction",
                    "IoU": round(max_gt_iou, 3),
                    "Environmental_Condition": condition,
                    "Rel_Area_Pct": geom["rel_area_pct"],
                    "Aspect_Ratio": geom["aspect_ratio"],
                    "Center_X_Norm": geom["cx_norm"],
                    "Center_Y_Norm": geom["cy_norm"],
                    "FP_Subtype": fp_subtype,
                    # Flagged if this prediction is already plotted on the Missed Target row
                    "Is_Duplicate_Candidate_Row": bool(pred["used_as_low_iou_candidate"]),
                })

        print(f"Finished {split_name}.")

    if not records:
        raise RuntimeError(
            f"Evaluation produced 0 records after checking {voc_devkit}. "
            f"Processed images: {processed_images}. No CSV was written."
        )

    out_file = base_dir / "voc_behavioral_results.csv"
    df = pd.DataFrame(records)
    df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"\nDone! Processed {len(df)} records across {processed_images} images. Saved to: {out_file}")


if __name__ == "__main__":
    main()