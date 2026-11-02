
import os, glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from xml.etree import ElementTree as ET
from ultralytics import YOLO

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

IMG_DIR = "VOC2007/JPEGImages"
ANN_DIR = "VOC2007/Annotations"
IMAGE_LIST = "VOC2007/ImageSets/Main/test.txt"

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


def run_inference(img_path, conf_thres=0.25):

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

# --- Matching (Prediction <-> Ground Truth) ----------------------------
def iou(a, b):
    """IoU zweier Boxen [xmin, ymin, xmax, ymax]."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def match_detections(gt_rows, det_df, iou_thres=0.5):
    """Ordnet jede Detection einer GT-Box zu (greedy nach Confidence):
    correct | wrong_class | missed."""
    out = []
    dets = det_df.sort_values("conf", ascending=False).to_dict("records")
    taken = set()
    for _, g in gt_rows.iterrows():
        gt_box = [g.xmin, g.ymin, g.xmax, g.ymax]
        best, best_iou = None, 0.0
        for i, d in enumerate(dets):
            if i in taken:
                continue
            ov = iou(gt_box, [d["xmin"], d["ymin"], d["xmax"], d["ymax"]])
            if ov > best_iou:
                best, best_iou = i, ov
        if best is not None and best_iou >= iou_thres:
            d = dets[best]; taken.add(best)
            status = "correct" if d["det_cls"] == g.cls else "wrong_class"
            out.append({**g.to_dict(), "status": status,
                        "det_cls": d["det_cls"], "conf": d["conf"], "det_iou": best_iou})
        else:
            out.append({**g.to_dict(), "status": "missed",
                        "det_cls": None, "conf": np.nan, "det_iou": best_iou})
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

phase1 = pd.concat(phase1_records, ignore_index=True)
phase1["status"] = pd.Categorical(
    phase1["status"], categories=["correct", "wrong_class", "missed"], ordered=True)
phase1.to_csv("phase1_results.csv", index=False)
print(phase1.groupby("status", observed=True).size())

# --- Visualisierungen (H1–H3) ----------------------------------
# H1: Erkennungsrate & Confidence nach Objektgröße
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
rate = phase1.assign(ok=phase1.status == "correct").groupby("size_bucket")["ok"].mean()
rate.reindex(["small", "medium", "large"]).plot.bar(ax=axes[0], color="#4C72B0")
axes[0].set_title("H1: Detection rate vs. object size")
axes[0].set_ylabel("correct-detection rate")

for bucket in ["small", "medium", "large"]:
    sub = phase1[(phase1.size_bucket == bucket) & phase1.conf.notna()]
    axes[1].hist(sub.conf, bins=20, alpha=0.5, label=bucket)
axes[1].set_title("H1: Confidence distribution per size")
axes[1].set_xlabel("confidence"); axes[1].legend()
plt.tight_layout(); plt.savefig("fig_h1_size.png", dpi=150); plt.show()

# H2: Performance vs. Bildkomplexität
fig, ax = plt.subplots(figsize=(6, 4))
comp = phase1.assign(ok=phase1.status == "correct").groupby("n_objects")["ok"].agg(["mean", "count"])
comp = comp[comp["count"] >= 10]  # seltenere Gruppen glätten
comp["mean"].plot(ax=ax, marker="o")
ax.set_title("H2: Detection rate vs. #objects per image")
ax.set_xlabel("objects per image"); ax.set_ylabel("correct-detection rate")
plt.tight_layout(); plt.savefig("fig_h2_complexity.png", dpi=150); plt.show()

# H3: Fehler nach Klasse (Confusion-Paare + Error-Rate pro Klasse)
conf_pairs = (phase1[phase1.status == "wrong_class"]
              .groupby(["cls", "det_cls"]).size().sort_values(ascending=False))
print("Top-10 class confusions:\n", conf_pairs.head(10))

err = phase1.assign(err=phase1.status != "correct").groupby("cls")["err"].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
err.plot.barh(ax=ax, color="#C44E52")
ax.set_title("H3: Error rate per class")
plt.tight_layout(); plt.savefig("fig_h3_classes.png", dpi=150); plt.show()

