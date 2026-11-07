
import os, glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from xml.etree import ElementTree as ET
from ultralytics import YOLO
from pathlib import Path
import seaborn as sns
from pathlib import Path

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED)

# VOC-Klassennamen (Reihenfolge wie in den VOC-Annotationen)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


VOC_TO_COCO_NAME = {
    "aeroplane": "airplane",       # umbenannt
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "diningtable": "dining table", # umbenannt
    "dog": "dog",
    "horse": "horse",
    "motorbike": "motorcycle",     # umbenannt
    "person": "person",
    "pottedplant": "potted plant", # umbenannt
    "sheep": "sheep",
    "sofa": "couch",               # in COCO "couch"!
    "train": "train",
    "tvmonitor": "tv",             # umbenannt
}

def load_model():


    model = YOLO("yolo26n.pt")
    coco_names = model.names  # dict: index -> name (80 COCO-Klassen)
    voc_to_coco_idx = {}
    for voc_name, coco_name in VOC_TO_COCO_NAME.items():
        matches = [i for i, n in coco_names.items() if n == coco_name]
        assert matches, f"COCO-Klasse '{coco_name}' nicht im Modell gefunden!"
        voc_to_coco_idx[voc_name] = matches[0]
    return model, voc_to_coco_idx

model, VOC_TO_COCO_IDX = load_model()
print(f"YOLO26n geladen, {len(VOC_TO_COCO_IDX)} VOC-Klassen gemappt.")


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "VOC2007"

IMG_DIR = str(DATASET_DIR / "JPEGImages")
ANN_DIR = str(DATASET_DIR / "Annotations")
IMAGE_LIST = str(DATASET_DIR / "ImageSets/Main/test.txt")

def load_voc_annotations(ann_dir):
    """Liest alle VOC-XML-Annotationen in einen DataFrame:
    image, cls, xmin, ymin, xmax, ymax, difficult."""
    records = []
    for xml_file in glob.glob(os.path.join(ann_dir, "*.xml")):
        root = ET.parse(xml_file).getroot()
        image_id = root.findtext("filename").rsplit(".", 1)[0]
        size = root.find("size")
        w, h = int(size.findtext("width")), int(size.findtext("height"))
        for obj in root.iter("object"):
            bbox = obj.find("bndbox")
            records.append({
                "image": image_id,
                "img_w": w, "img_h": h,
                "cls": obj.findtext("name"),
                "difficult": int(obj.findtext("difficult")),
                "xmin": float(bbox.findtext("xmin")),
                "ymin": float(bbox.findtext("ymin")),
                "xmax": float(bbox.findtext("xmax")),
                "ymax": float(bbox.findtext("ymax")),
            })
    return pd.DataFrame(records)

gt = load_voc_annotations(ANN_DIR)

# Abgeleitete Eigenschaften pro Objekt (für H1, H2, H3):
#gt["area"] = (gt.xmax - gt.xmin) * (gt.ymax - gt.ymin)
gt["area"] = (
    (gt["xmax"] - gt["xmin"]) *
    (gt["ymax"] - gt["ymin"])
)


gt["rel_area"] = gt.area / (gt.img_w * gt.img_h)          # Objektgröße relativ zum Bild (H1)
gt["rel_w"] = (gt.xmax - gt.xmin) / gt.img_w
gt["rel_h"] = (gt.ymax - gt.ymin) / gt.img_h

# Bildkomplexität (H2): Anzahl Objekte pro Bild
gt["n_objects"] = gt.groupby("image")["cls"].transform("count")

# Größenkategorien (COCO-Konvention, angepasst an VOC-Auflösung ~500px)
def size_bucket(rel_w, rel_h):
    a = rel_w * rel_h
    if a < 32**2 / (500 * 500):  return "small"
    if a < 96**2 / (500 * 500):  return "medium"
    return "large"

gt["size_bucket"] = gt.apply(lambda r: size_bucket(r.rel_w, r.rel_h), axis=1)
print(gt.groupby("size_bucket")["cls"].count())
print(gt.head())


CLASS_TYPES = {
    "Human": ["person"],
    "Animal": [
        "bird", "cat", "cow", "dog",
        "horse", "sheep"
    ],
    "Vehicle": [
        "aeroplane", "bicycle", "boat",
        "bus", "car", "motorbike", "train"
    ],
    "Object": [
        "bottle", "chair", "diningtable",
        "pottedplant", "sofa", "tvmonitor"
    ]
}

CLASS_TO_TYPE = {
    cls: group
    for group, classes in CLASS_TYPES.items()
    for cls in classes
}

gt["class_type"] = gt["cls"].map(CLASS_TO_TYPE)

# Normalized center coordinates: 0 to 1
gt["center_x"] = (
    (gt["xmin"] + gt["xmax"]) / 2 / gt["img_w"]
)
gt["center_y"] = (
    (gt["ymin"] + gt["ymax"]) / 2 / gt["img_h"]
)

# 0 = image center, 1 = image corner
gt["centricity"] = np.sqrt(
    (gt["center_x"] - 0.5)**2 +
    (gt["center_y"] - 0.5)**2
) / np.sqrt(0.5)

# Ground-truth bounding-box overlap
gt["max_gt_iou"] = 0.0
gt["overlap_ratio"] = 0.0
gt["overlapped"] = False

for image_id, group in gt.groupby("image"):
    indices = group.index.to_list()

    for i in indices:
        a = gt.loc[i, ["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
        area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])

        max_iou = 0.0
        max_overlap_ratio = 0.0

        for j in indices:
            if i == j:
                continue

            b = gt.loc[j, ["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)

            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])

            intersection = max(0, x2-x1) * max(0, y2-y1)
            area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
            union = area_a + area_b - intersection

            box_iou = intersection / union if union > 0 else 0
            overlap_ratio = intersection / area_a if area_a > 0 else 0

            max_iou = max(max_iou, box_iou)
            max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

        gt.loc[i, "max_gt_iou"] = max_iou
        gt.loc[i, "overlap_ratio"] = max_overlap_ratio
        gt.loc[i, "overlapped"] = max_overlap_ratio > 0

print(gt[[
    "cls", "class_type", "rel_area",
    "centricity", "max_gt_iou",
    "overlap_ratio", "overlapped"
]].head())

def run_inference(img_path, conf_thres=0.01):

    coco_to_voc = {coco_idx: voc_name for voc_name, coco_idx in VOC_TO_COCO_IDX.items()}
    results = model(img_path, conf=conf_thres, verbose=False)
    r = results[0]
    dets = []
    for box in r.boxes:
        coco_idx = int(box.cls)
        if coco_idx not in coco_to_voc:
            continue  # COCO-Klasse außerhalb der 20 VOC-Klassen (z.B. "backpack")
        dets.append({
            "det_cls": coco_to_voc[coco_idx],  # zurückübersetzt in VOC-Namen
            "conf": float(box.conf),
            "xmin": float(box.xyxy[0][0]), "ymin": float(box.xyxy[0][1]),
            "xmax": float(box.xyxy[0][2]), "ymax": float(box.xyxy[0][3]),
        })
   ## return pd.DataFrame(dets), r
    return pd.DataFrame(
        dets,
        columns=["det_cls", "conf", "xmin", "ymin", "xmax", "ymax"]
    ), r

# Matching (Prediction <-> Ground Truth)
def iou(a, b):
    """IoU zweier Boxen [xmin, ymin, xmax, ymax]."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_detections(gt_rows, det_df, iou_thres=0.5):
    gt_list = gt_rows.to_dict("records")
    det_list = det_df.to_dict("records")

    matches = {}
    used_gt = set()
    used_det = set()

    candidates = []

    for gi, g in enumerate(gt_list):
        gb = [g[k] for k in ["xmin", "ymin", "xmax", "ymax"]]

        for di, d in enumerate(det_list):
            db = [d[k] for k in ["xmin", "ymin", "xmax", "ymax"]]
            ov = iou(gb, db)

            if ov >= iou_thres:
                candidates.append((
                    gi, di, ov,
                    g["cls"] == d["det_cls"],
                    d["conf"]
                ))


    for require_correct in [True, False]:
        eligible = [
            c for c in candidates
            if c[3] == require_correct
        ]

        eligible.sort(
            key=lambda c: (c[2], c[4]),
            reverse=True
        )

        for gi, di, ov, same_class, conf in eligible:
            if gi in used_gt or di in used_det:
                continue

            matches[gi] = (di, ov)
            used_gt.add(gi)
            used_det.add(di)

    out = []

    for gi, g in enumerate(gt_list):
        gb = [g[k] for k in ["xmin", "ymin", "xmax", "ymax"]]

        candidate_pairs = [
            (iou(gb, [d[k] for k in ["xmin", "ymin", "xmax", "ymax"]]), di)
            for di, d in enumerate(det_list)
        ]
        best_ov, best_di = max(candidate_pairs, default=(np.nan, None))
        candidate_info = {
            "candidate_iou": best_ov,
            "candidate_conf": det_list[best_di]["conf"] if best_di is not None else np.nan,
            "candidate_cls": det_list[best_di]["det_cls"] if best_di is not None else None,
            "has_candidate": best_di is not None,
        }
        if gi in matches:
            di, ov = matches[gi]
            d = det_list[di]

            status = (
                "correct"
                if g["cls"] == d["det_cls"]
                else "wrong_class"
            )

            out.append({
                **g,
                "status": status,
                "det_cls": d["det_cls"],
                "conf": d["conf"],
                "det_iou": ov,
                **candidate_info
            })

        else:
            gb = [g[k] for k in ["xmin", "ymin", "xmax", "ymax"]]

            best_iou = max(
                (
                    iou(
                        gb,
                        [d[k] for k in
                         ["xmin", "ymin", "xmax", "ymax"]]
                    )
                    for d in det_list
                ),
                default=0.0
            )

            out.append({
                **g,
                "status": "missed",
                "det_cls": None,
                "conf": np.nan,
                "det_iou": np.nan,
                "best_any_iou": best_iou,
                **candidate_info
            })

    return pd.DataFrame(out)


#  Phase I über den Test-Loop laufen lassen --------------------------
image_ids = [line.strip() for line in open(IMAGE_LIST)]
MAX_IMAGES = 500


phase1_records = []
for img_id in image_ids[:MAX_IMAGES]:
    img_path = os.path.join(IMG_DIR, img_id + ".jpg")
    det_df, _ = run_inference(img_path)
    gt_rows = gt[gt.image == img_id]
    matched = match_detections(gt_rows, det_df)
    matched["image_path"] = img_path
    phase1_records.append(matched)

BASE_DIR = Path(__file__).resolve().parent

phase1 = pd.concat(phase1_records, ignore_index=True)

phase1["status"] = pd.Categorical(
    phase1["status"],
    categories=["correct", "wrong_class", "missed"],
    ordered=True
)

# Save CSV
phase1.to_csv(BASE_DIR / "phase1_results.csv", index=False)

print(phase1.groupby("status", observed=True).size())

