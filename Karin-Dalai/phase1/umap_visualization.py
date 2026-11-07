from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import models
from sklearn.preprocessing import normalize
import umap

# Configuration
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "phase1_results.csv"
IMG_DIR = ROOT / "VOC2007" / "JPEGImages"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

BATCH_SIZE = 32
SEED = 42

CLASS_TYPES = {
    "Human": ["person"],
    "Animal": ["bird", "cat", "cow", "dog", "horse", "sheep"],
    "Vehicle": ["aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train"],
    "Object": ["bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]
}

CLASS_TO_TYPE = {
    cls: group
    for group, classes in CLASS_TYPES.items()
    for cls in classes
}

# Load data
df = pd.read_csv(CSV_PATH, dtype={"image": str})
df["image"] = df["image"].str.strip().str.zfill(6)

# Load pretrained ResNet18
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

weights = models.ResNet18_Weights.DEFAULT
encoder = models.resnet18(weights=weights)
encoder.fc = torch.nn.Identity()
encoder = encoder.to(device).eval()
preprocess = weights.transforms()

# Extract object features
features = []
valid_indices = []
batch = []

def extract_batch():
    if not batch:
        return
    with torch.inference_mode():
        inputs = torch.stack(batch).to(device)
        outputs = encoder(inputs).cpu().numpy()
    features.extend(outputs)
    batch.clear()

for idx, row in df.iterrows():
    path = IMG_DIR / f"{row['image']}.jpg"

    if not path.exists():
        continue

    with Image.open(path) as im:
        im = im.convert("RGB")

        box = tuple(
            int(row[k])
            for k in ["xmin", "ymin", "xmax", "ymax"]
        )

        crop = im.crop(box)

        if crop.width < 2 or crop.height < 2:
            continue

        batch.append(preprocess(crop))
        valid_indices.append(idx)

        if len(batch) == BATCH_SIZE:
            extract_batch()

extract_batch()

# Normalize feature vectors
X = normalize(np.array(features, dtype=np.float32))
df = df.loc[valid_indices].reset_index(drop=True)

# UMAP
reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=SEED
)

coords = reducer.fit_transform(X)
df["umap_x"] = coords[:, 0]
df["umap_y"] = coords[:, 1]
df["class_type"] = df["cls"].map(CLASS_TO_TYPE)

# Umap data
df.to_csv(OUT / "umap_results.csv", index=False)

# Visualization 1: Semantic categories
sns.set_theme(style="white")

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x="umap_x",
    y="umap_y",
    hue="class_type",
    hue_order=["Human", "Animal", "Vehicle", "Object"],
    s=25,
    alpha=0.7,
    linewidth=0
)

plt.title("UMAP: Visual Similarity by Object Category")
plt.tight_layout()
plt.savefig(OUT / "1_umap_class_type.png", dpi=200)
plt.close()

# Visualization 2: Detection status
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x="umap_x",
    y="umap_y",
    hue="status",
    palette={
        "correct": "#2ca02c",
        "missed": "#d62728",
        "wrong_class": "#ff7f0e"
    },
    s=25,
    alpha=0.7,
    linewidth=0
)

plt.title("YOLO26n Detection Status in UMAP Space")
plt.savefig(OUT / "2_umap_detection_status.png", dpi=200)
plt.close()

print("Successfully generated UMAP visualizations.")