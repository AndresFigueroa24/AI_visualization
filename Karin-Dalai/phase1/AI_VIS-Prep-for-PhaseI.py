from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "VOC2007"
IMG_DIR = DATASET_DIR / "JPEGImages"
ANN_DIR = DATASET_DIR / "Annotations"
IMAGE_LIST = DATASET_DIR / "ImageSets" / "Main" / "test.txt"
MAX_IMAGES = 500
CONF_THRESHOLD = 0.01
IOU_THRESHOLD = 0.5

CLASS_TYPES = {
    "Human": ["person"],
    "Animal": ["bird", "cat", "cow", "dog", "horse", "sheep"],
    "Vehicle": ["aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train"],
    "Object": ["bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"],
}
CLASS_TO_TYPE = {
    cls: class_type
    for class_type, classes in CLASS_TYPES.items()
    for cls in classes
}
VOC_TO_COCO_NAME = {cls: cls for cls in CLASS_TO_TYPE}
VOC_TO_COCO_NAME.update({
    "aeroplane": "airplane",
    "diningtable": "dining table",
    "motorbike": "motorcycle",
    "pottedplant": "potted plant",
    "sofa": "couch",
    "tvmonitor": "tv",
})
BOX_COLUMNS = ["xmin", "ymin", "xmax", "ymax"]
STATUSES = ["correct", "wrong_class", "missed"]


def load_annotations(image_ids):
    records = []
    for image_id in image_ids:
        root = ET.parse(ANN_DIR / f"{image_id}.xml").getroot()
        width = int(root.findtext("size/width"))
        height = int(root.findtext("size/height"))
        for obj in root.findall("object"):
            box = obj.find("bndbox")
            records.append({
                "image": image_id,
                "img_w": width,
                "img_h": height,
                "cls": obj.findtext("name"),
                "difficult": int(obj.findtext("difficult", "0")),
                **{name: float(box.findtext(name)) for name in BOX_COLUMNS},
            })
    return pd.DataFrame(records)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def prepare_annotations(gt):
    gt = gt.copy()
    gt["area"] = (gt.xmax - gt.xmin) * (gt.ymax - gt.ymin)
    gt["rel_area"] = gt.area / (gt.img_w * gt.img_h)
    gt["rel_w"] = (gt.xmax - gt.xmin) / gt.img_w
    gt["rel_h"] = (gt.ymax - gt.ymin) / gt.img_h
    gt["n_objects"] = gt.groupby("image")["cls"].transform("size")
    gt["size_bucket"] = pd.cut(
        gt.rel_area,
        bins=[-np.inf, 32**2 / 500**2, 96**2 / 500**2, np.inf],
        labels=["small", "medium", "large"],
        right=False,
    ).astype(str)
    gt["class_type"] = gt.cls.map(CLASS_TO_TYPE)
    gt["center_x"] = (gt.xmin + gt.xmax) / (2 * gt.img_w)
    gt["center_y"] = (gt.ymin + gt.ymax) / (2 * gt.img_h)
    gt["centricity"] = np.hypot(gt.center_x - 0.5, gt.center_y - 0.5) / np.sqrt(0.5)
    gt["max_gt_iou"] = 0.0
    gt["overlap_ratio"] = 0.0

    for _, group in gt.groupby("image"):
        boxes = group[BOX_COLUMNS].to_numpy()
        areas = group.area.to_numpy()
        max_iou = np.zeros(len(group))
        max_overlap = np.zeros(len(group))
        for i, a in enumerate(boxes):
            for j, b in enumerate(boxes):
                if i == j:
                    continue
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0, min(a[3], b[3]) - max(a[1], b[1])
                )
                union = areas[i] + areas[j] - intersection
                max_iou[i] = max(max_iou[i], intersection / union if union > 0 else 0)
                max_overlap[i] = max(max_overlap[i], intersection / areas[i] if areas[i] > 0 else 0)
        gt.loc[group.index, "max_gt_iou"] = max_iou
        gt.loc[group.index, "overlap_ratio"] = max_overlap

    gt["overlapped"] = gt.overlap_ratio > 0
    return gt


def run_inference(model, image_path, coco_to_voc):
    result = model(str(image_path), conf=CONF_THRESHOLD, verbose=False)[0]
    detections = []
    for box in result.boxes:
        cls = coco_to_voc.get(int(box.cls.item()))
        if cls is None:
            continue
        detections.append({
            "det_cls": cls,
            "conf": float(box.conf.item()),
            **dict(zip(BOX_COLUMNS, box.xyxy[0].tolist())),
        })
    return detections


def match_detections(gt_rows, detections):
    ground_truth = gt_rows.to_dict("records")
    overlaps = [
        [iou([g[k] for k in BOX_COLUMNS], [d[k] for k in BOX_COLUMNS]) for d in detections]
        for g in ground_truth
    ]
    candidates = [
        (gi, di, ov, g["cls"] == detections[di]["det_cls"], detections[di]["conf"])
        for gi, g in enumerate(ground_truth)
        for di, ov in enumerate(overlaps[gi])
        if ov >= IOU_THRESHOLD
    ]
    matches = {}
    used_detections = set()

    for same_class in (True, False):
        eligible = sorted(
            (candidate for candidate in candidates if candidate[3] == same_class),
            key=lambda candidate: (candidate[2], candidate[4]),
            reverse=True,
        )
        for gi, di, overlap, _, _ in eligible:
            if gi not in matches and di not in used_detections:
                matches[gi] = (di, overlap)
                used_detections.add(di)

    records = []
    for gi, g in enumerate(ground_truth):
        scores = overlaps[gi]
        best_di = max(range(len(scores)), key=lambda di: (scores[di], di)) if scores else None
        candidate = detections[best_di] if best_di is not None else None
        row = {
            **g,
            "candidate_iou": scores[best_di] if candidate is not None else np.nan,
            "candidate_conf": candidate["conf"] if candidate is not None else np.nan,
            "candidate_cls": candidate["det_cls"] if candidate is not None else None,
            "has_candidate": candidate is not None,
        }
        if gi in matches:
            di, overlap = matches[gi]
            detection = detections[di]
            row.update({
                "status": "correct" if g["cls"] == detection["det_cls"] else "wrong_class",
                "det_cls": detection["det_cls"],
                "conf": detection["conf"],
                "det_iou": overlap,
                "best_any_iou": np.nan,
            })
        else:
            row.update({
                "status": "missed",
                "det_cls": None,
                "conf": np.nan,
                "det_iou": np.nan,
                "best_any_iou": max(scores, default=0.0),
            })
        records.append(row)
    return records


def main():
    image_ids = IMAGE_LIST.read_text().splitlines()[:MAX_IMAGES]
    gt = prepare_annotations(load_annotations(image_ids))
    model = YOLO("yolo26n.pt")
    coco_to_voc = {
        index: voc
        for index, name in model.names.items()
        for voc, coco_name in VOC_TO_COCO_NAME.items()
        if name == coco_name
    }

    records = []
    for image_id, gt_rows in gt.groupby("image", sort=False):
        image_path = IMG_DIR / f"{image_id}.jpg"
        detections = run_inference(model, image_path, coco_to_voc)
        for row in match_detections(gt_rows, detections):
            row["image_path"] = str(image_path)
            records.append(row)

    phase1 = pd.DataFrame(records)
    phase1["status"] = pd.Categorical(phase1.status, categories=STATUSES, ordered=True)
    phase1.to_csv(ROOT / "phase1_results.csv", index=False)
    print(phase1.groupby("status", observed=True).size())


if __name__ == "__main__":
    main()
