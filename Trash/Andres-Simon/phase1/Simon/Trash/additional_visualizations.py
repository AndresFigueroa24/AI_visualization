from __future__ import annotations

"""
VIS4AI Phase I — Lecture Concepts Explorer for YOLO / PASCAL VOC.

Save this file as vis4ai_lecture_explorer.py next to
voc_behavioral_results.csv and start it with:

    streamlit run vis4ai_lecture_explorer.py

Optional VOC images/annotations are searched under:
    VOC_dataset/VOCdevkit
    archive/VOC_dataset/VOCdevkit
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from PIL import Image, ImageDraw

from scipy.cluster.hierarchy import leaves_list, linkage

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "voc_behavioral_results.csv"

VOC_CLASSES_SEMANTIC = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

VOC_CLASSES_ALPHABETICAL = sorted(VOC_CLASSES_SEMANTIC)

BACKGROUND_LABEL = "None (Background)"

SUPER_CATEGORY_MAP = {
    "aeroplane": "Vehicles",
    "bicycle": "Vehicles",
    "boat": "Vehicles",
    "bus": "Vehicles",
    "car": "Vehicles",
    "motorbike": "Vehicles",
    "train": "Vehicles",

    "bird": "Animals & Person",
    "cat": "Animals & Person",
    "cow": "Animals & Person",
    "dog": "Animals & Person",
    "horse": "Animals & Person",
    "person": "Animals & Person",
    "sheep": "Animals & Person",

    "bottle": "Indoor Objects",
    "chair": "Indoor Objects",
    "diningtable": "Indoor Objects",
    "pottedplant": "Indoor Objects",
    "sofa": "Indoor Objects",
    "tvmonitor": "Indoor Objects",
}

SUPER_COLORS = {
    "Animals & Person": "#2563eb",
    "Vehicles": "#16a34a",
    "Indoor Objects": "#f59e0b",
    "Background": "#94a3b8",
}

OUTCOME_COLORS_5 = {
    "TP (Conf, IoU>T)": "#2ca02c",
    "Misclassification": "#d62728",
    "Missed (IoU<T)": "#10b981",
    "Missed (Conf<T)": "#3b82f6",
    "Blind Spot (IoU=0)": "#475569",
    "FP (Hallucination)": "#ff7f0e",
    "Filtered FP": "#94a3b8",
}

VOC_NAME_MAP = {
    "airplane": "aeroplane",
    "motorcycle": "motorbike",
    "couch": "sofa",
    "dining table": "diningtable",
    "potted plant": "pottedplant",
    "tv": "tvmonitor",
    "television": "tvmonitor",
}


# =============================================================================
# GENERAL HELPERS AND DATA LOADING
# =============================================================================

def normalize_label(name: object) -> str:
    """Normalize labels from CSV/XML and common COCO/VOC naming variants."""

    cleaned = str(name).strip().lower()

    if cleaned in {
        "none (background)",
        "none(background)",
        "background",
    }:
        return BACKGROUND_LABEL

    if cleaned in {"none", "null", "nan", ""}:
        return "None"

    if cleaned in VOC_NAME_MAP:
        return VOC_NAME_MAP[cleaned]

    return cleaned.replace(" ", "").replace("_", "").replace("-", "")


def locate_voc_devkit() -> Path | None:
    """Find VOCdevkit in the project folder or common archive subfolder."""

    candidates = [
        BASE_DIR / "VOC_dataset" / "VOCdevkit",
        BASE_DIR / "archive" / "VOC_dataset" / "VOCdevkit",
        BASE_DIR / "VOCdevkit",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return None


def _sample_parts(sample_id: object) -> tuple[str | None, str]:
    """Split IDs such as VOC2007_000001 into (VOC2007, 000001)."""

    value = str(sample_id).strip()

    for split_name in ("VOC2007", "VOC2012"):
        prefix = split_name + "_"

        if value.startswith(prefix):
            return split_name, value[len(prefix):]

    if "_" in value:
        left, right = value.split("_", 1)

        if left.upper().startswith("VOC"):
            return left, right

    return None, value


def _read_xml_geometry(devkit: Path) -> tuple[dict, dict]:
    """Return per-image boxes and object counts read from VOC XML annotations."""

    geometry_by_image = {}
    count_by_image = {}

    for split_name in ("VOC2007", "VOC2012"):

        ann_dir = devkit / split_name / "Annotations"

        if not ann_dir.exists():
            continue

        for xml_file in ann_dir.glob("*.xml"):

            sample_id = f"{split_name}_{xml_file.stem}"

            try:
                root = ET.parse(xml_file).getroot()

                size_node = root.find("size")

                if size_node is None:
                    continue

                width_node = size_node.find("width")
                height_node = size_node.find("height")

                if width_node is None or height_node is None:
                    continue

                image_width = max(1.0, float(width_node.text))
                image_height = max(1.0, float(height_node.text))

                image_area = image_width * image_height

                class_boxes = {}
                object_count = 0

                for obj in root.findall("object"):

                    name_node = obj.find("name")
                    box_node = obj.find("bndbox")

                    if name_node is None or box_node is None:
                        continue

                    coords = [
                        box_node.find(key)
                        for key in ("xmin", "ymin", "xmax", "ymax")
                    ]

                    if any(node is None for node in coords):
                        continue

                    class_name = normalize_label(name_node.text)

                    xmin, ymin, xmax, ymax = [
                        float(node.text) for node in coords
                    ]

                    box_width = max(1.0, xmax - xmin)
                    box_height = max(1.0, ymax - ymin)

                    rel_area = (
                        box_width * box_height / image_area
                    ) * 100.0

                    aspect_ratio = box_width / box_height

                    center_x = (
                        (xmin + xmax) * 0.5
                    ) / image_width

                    center_y = (
                        (ymin + ymax) * 0.5
                    ) / image_height

                    class_boxes.setdefault(
                        class_name, []
                    ).append(
                        (rel_area, aspect_ratio, center_x, center_y)
                    )

                    object_count += 1

                geometry_by_image[sample_id] = class_boxes
                count_by_image[sample_id] = object_count

            except (ET.ParseError, OSError, TypeError, ValueError):
                continue

    return geometry_by_image, count_by_image


@st.cache_data(show_spinner="Loading and enriching VOC results…")
def load_and_enrich_data() -> pd.DataFrame:
    """
    Load the behavioral CSV and add interpretable per-instance attributes.

    Geometry is read from VOC XML files wherever possible.

    If XML geometry is unavailable, a deterministic fallback uses the
    scale category from the CSV. Such fallback values are explicitly marked.
    """

    if not CSV_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(CSV_PATH)

    if df.empty:
        return df

    required_defaults = {
        "Sample_ID": "unknown_sample",
        "Outcome": "Unknown",
        "Predicted_Class": "None",
        "Ground_Truth": BACKGROUND_LABEL,
        "Confidence": np.nan,
        "IoU": 0.0,
        "Environmental_Condition": "Unknown",
        "Split": "Unknown",
    }

    for column, default_value in required_defaults.items():

        if column not in df.columns:
            df[column] = default_value

    df["Sample_ID"] = (
        df["Sample_ID"].fillna("unknown_sample").astype(str)
    )

    df["Outcome"] = (
        df["Outcome"].fillna("Unknown").astype(str)
    )

    df["Predicted_Class"] = (
        df["Predicted_Class"].fillna("None").map(normalize_label)
    )

    df["Ground_Truth"] = (
        df["Ground_Truth"].fillna(BACKGROUND_LABEL).map(normalize_label)
    )

    df.loc[
        df["Ground_Truth"].isin(
            ["none", "background", "none(background)"]
        ),
        "Ground_Truth"
    ] = BACKGROUND_LABEL

    df["Confidence"] = pd.to_numeric(
        df["Confidence"], errors="coerce"
    )

    df["IoU"] = pd.to_numeric(
        df["IoU"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)

    if "Record_ID" not in df.columns:

        df["Record_ID"] = [
            f"rec_{i:05d}" for i in range(len(df))
        ]

    if "Candidate_Class" not in df.columns:

        df["Candidate_Class"] = np.where(
            df["Outcome"].eq("Missed Target"),
            "None",
            df["Predicted_Class"]
        )

    else:

        df["Candidate_Class"] = (
            df["Candidate_Class"].fillna("None").map(normalize_label)
        )

    if "Candidate_Confidence" not in df.columns:

        df["Candidate_Confidence"] = df["Confidence"]

    else:

        df["Candidate_Confidence"] = pd.to_numeric(
            df["Candidate_Confidence"],
            errors="coerce"
        )

    # A zero-IoU missed target has no usable localized candidate confidence.

    blind_mask = (
        df["Outcome"].eq("Missed Target")
        & df["IoU"].le(0.0)
    )

    df.loc[blind_mask, "Confidence"] = np.nan
    df.loc[blind_mask, "Candidate_Confidence"] = np.nan

    # Effective target class for grouping.

    df["Target_Class"] = np.where(
        df["Ground_Truth"].ne(BACKGROUND_LABEL),
        df["Ground_Truth"],
        df["Predicted_Class"]
    )

    df["Super_Category"] = (
        df["Target_Class"]
        .map(SUPER_CATEGORY_MAP)
        .fillna("Background")
    )

    if "Is_Duplicate_Candidate_Row" not in df.columns:
        df["Is_Duplicate_Candidate_Row"] = False

    duplicate_values = df["Is_Duplicate_Candidate_Row"]

    if duplicate_values.dtype == object:

        df["Is_Duplicate_Candidate_Row"] = (
            duplicate_values.fillna("false")
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )

    else:

        df["Is_Duplicate_Candidate_Row"] = (
            duplicate_values.fillna(False).astype(bool)
        )

    geometry_columns = [
        "Rel_Area_Pct",
        "Aspect_Ratio",
        "Center_X_Norm",
        "Center_Y_Norm",
    ]

    for column in geometry_columns:

        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    devkit = locate_voc_devkit()

    xml_geometry = {}
    xml_counts = {}

    if devkit is not None:

        xml_geometry, xml_counts = _read_xml_geometry(devkit)

    # Match CSV rows to annotation boxes by image and target class.
    # Repeated objects of the same class are assigned in deterministic order.

    used_tracker = {}

    rng = np.random.default_rng(seed=42)

    geometry_sources = []

    for row_index, row in df.iterrows():

        sample_id = str(row["Sample_ID"])
        target_class = normalize_label(row["Target_Class"])

        key = (sample_id, target_class)

        candidates = (
            xml_geometry.get(sample_id, {}).get(target_class, [])
        )

        candidate_index = used_tracker.get(key, 0)

        xml_values = (
            candidates[candidate_index]
            if candidate_index < len(candidates)
            else None
        )

        if xml_values is not None:

            used_tracker[key] = candidate_index + 1

            (
                rel_area,
                aspect_ratio,
                center_x,
                center_y,
            ) = xml_values

            source = "VOC XML"

        else:

            rel_area = row["Rel_Area_Pct"]
            aspect_ratio = row["Aspect_Ratio"]
            center_x = row["Center_X_Norm"]
            center_y = row["Center_Y_Norm"]

            source = "CSV"

            # Preserve supplied CSV geometry.
            # Fill only unavailable values.

            if (
                pd.isna(rel_area)
                or pd.isna(aspect_ratio)
                or pd.isna(center_x)
                or pd.isna(center_y)
            ):

                scale_label = str(
                    row["Environmental_Condition"]
                ).lower()

                if "small" in scale_label:

                    fallback_area = float(
                        rng.uniform(0.5, 3.8)
                    )

                elif "medium" in scale_label:

                    fallback_area = float(
                        rng.uniform(4.2, 19.5)
                    )

                else:

                    fallback_area = float(
                        rng.uniform(20.5, 68.0)
                    )

                if pd.isna(rel_area):
                    rel_area = fallback_area

                if pd.isna(aspect_ratio):

                    aspect_ratio = float(
                        np.exp(rng.normal(0.0, 0.45))
                    )

                if pd.isna(center_x):

                    center_x = float(
                        np.clip(
                            rng.normal(0.5, 0.22),
                            0.05,
                            0.95
                        )
                    )

                if pd.isna(center_y):

                    center_y = float(
                        np.clip(
                            rng.normal(0.5, 0.22),
                            0.05,
                            0.95
                        )
                    )

                source = "Synthetic fallback (no XML/CSV geometry)"

        df.at[row_index, "Rel_Area_Pct"] = float(rel_area)
        df.at[row_index, "Aspect_Ratio"] = float(aspect_ratio)
        df.at[row_index, "Center_X_Norm"] = float(center_x)
        df.at[row_index, "Center_Y_Norm"] = float(center_y)

        geometry_sources.append(source)

    df["Geometry_Source"] = geometry_sources

    # Normalized radial distance from image center.

    df["Center_Dist_Norm"] = np.sqrt(
        (df["Center_X_Norm"] - 0.5) ** 2
        + (df["Center_Y_Norm"] - 0.5) ** 2
    ) / np.sqrt(0.5)

    df["Center_Dist_Norm"] = (
        df["Center_Dist_Norm"].clip(0.0, 1.0).round(3)
    )

    # Prefer exact XML object counts.
    # Otherwise use a row-count proxy from the CSV.

    row_gt_counts = (
        df.loc[
            df["Ground_Truth"].ne(BACKGROUND_LABEL)
        ]
        .groupby("Sample_ID")["Record_ID"]
        .count()
        .to_dict()
    )

    row_total_counts = (
        df.groupby("Sample_ID")["Record_ID"]
        .count()
        .to_dict()
    )

    df["Scene_GT_Objects"] = df["Sample_ID"].map(
        lambda sid: int(
            xml_counts.get(
                sid,
                row_gt_counts.get(sid, 1)
            )
        )
    )

    df["Scene_Total_Boxes"] = df["Sample_ID"].map(
        lambda sid: int(
            row_total_counts.get(sid, 1)
        )
    )

    df["Scene_GT_Objects"] = (
        df["Scene_GT_Objects"].clip(lower=1)
    )

    df["Scene_Total_Boxes"] = (
        df["Scene_Total_Boxes"].clip(lower=1)
    )

    return df


# =============================================================================
# THRESHOLD-DEPENDENT CLASSIFICATION STATE
# =============================================================================

def evaluate_dynamic_state(
    df_in: pd.DataFrame,
    conf_t: float,
    iou_t: float
) -> pd.DataFrame:

    """Re-evaluate each row for a chosen confidence and IoU threshold pair."""

    d = df_in.copy()

    conf = pd.to_numeric(
        d["Candidate_Confidence"],
        errors="coerce"
    ).to_numpy(dtype=float)

    iou = pd.to_numeric(
        d["IoU"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    original_outcome = d["Outcome"].astype(str).to_numpy()

    gt_class = d["Ground_Truth"].astype(str).to_numpy()

    candidate_class = (
        d["Candidate_Class"].fillna("None").astype(str).to_numpy()
    )

    predicted_class = (
        d["Predicted_Class"].fillna("None").astype(str).to_numpy()
    )

    effective_class = np.where(
        candidate_class != "None",
        candidate_class,
        predicted_class
    )

    is_ground_truth = (
        (gt_class != BACKGROUND_LABEL)
        & (original_outcome != "False Positive")
    )

    is_false_positive = ~is_ground_truth

    has_candidate = (
        np.isfinite(conf) & (conf >= 0.0)
    )

    passes_confidence = (
        has_candidate & (conf >= conf_t)
    )

    passes_iou = iou >= iou_t

    has_class = effective_class != "None"

    correct_class = effective_class == gt_class

    dynamic_outcome = np.full(
        len(d),
        "Blind Spot (IoU=0)",
        dtype=object
    )

    # Ground-truth objects.

    blind = (
        is_ground_truth
        & (~has_candidate | (iou <= 0.0))
    )

    low_conf = (
        is_ground_truth
        & has_candidate
        & (iou > 0.0)
        & (~passes_confidence)
    )

    active_gt = (
        is_ground_truth & passes_confidence
    )

    active_tp = (
        active_gt
        & passes_iou
        & has_class
        & correct_class
    )

    active_mis = (
        active_gt
        & passes_iou
        & has_class
        & (~correct_class)
    )

    active_loc_miss = (
        active_gt
        & (~(passes_iou & has_class))
    )

    dynamic_outcome[blind] = "Blind Spot (IoU=0)"

    dynamic_outcome[low_conf] = "Missed (Conf<T)"

    dynamic_outcome[active_tp] = "TP (Conf, IoU>T)"

    dynamic_outcome[active_mis] = "Misclassification"

    dynamic_outcome[active_loc_miss] = "Missed (IoU<T)"

    # Background candidates.

    accepted_fp = (
        is_false_positive & passes_confidence
    )

    filtered_fp = (
        is_false_positive & (~passes_confidence)
    )

    dynamic_outcome[accepted_fp] = "FP (Hallucination)"

    dynamic_outcome[filtered_fp] = "Filtered FP"

    d["Dynamic_Outcome"] = dynamic_outcome

    d["Effective_Pred_Class"] = effective_class

    return d


def classify_conceptual_confusion(
    gt_cls: str,
    pred_cls: str
) -> str:

    """Assign a lecture-inspired conceptual label to a class confusion."""

    gt_cls = normalize_label(gt_cls)
    pred_cls = normalize_label(pred_cls)

    if pred_cls not in VOC_CLASSES_SEMANTIC:

        return "4. Taxonomy Shift / COCO OOV (80 → 20 Classes)"

    pair = frozenset((gt_cls, pred_cls))

    co_occurrence_pairs = {
        frozenset(pair)
        for pair in [
            ("person", "chair"),
            ("person", "sofa"),
            ("person", "bicycle"),
            ("person", "motorbike"),
            ("person", "horse"),
            ("person", "diningtable"),
            ("person", "car"),
            ("person", "bus"),
            ("person", "boat"),
            ("chair", "diningtable"),
            ("bottle", "diningtable"),
            ("pottedplant", "chair"),
            ("pottedplant", "sofa"),
            ("tvmonitor", "sofa"),
            ("tvmonitor", "chair"),
        ]
    }

    if pair in co_occurrence_pairs:

        return "2. Part-of / Scene Co-Occurrence Relationship"

    if (
        gt_cls in SUPER_CATEGORY_MAP
        and pred_cls in SUPER_CATEGORY_MAP
        and SUPER_CATEGORY_MAP[gt_cls]
        == SUPER_CATEGORY_MAP[pred_cls]
    ):

        return "1. Subclass / Intra-Family Sibling Relationship"

    return "3. Cross-Category Visual Ambiguity"


# =============================================================================
# VOC SAMPLE IMAGE DRILL-DOWN
# =============================================================================

def load_voc_sample_with_gt_boxes(
    sample_id: str,
    highlight_cls: str | None = None
) -> Image.Image | None:

    """Load the original VOC image and draw its ground-truth annotation boxes."""

    devkit = locate_voc_devkit()

    if devkit is None:
        return None

    split_name, stem = _sample_parts(sample_id)

    candidate_splits = (
        [split_name]
        if split_name
        else ["VOC2007", "VOC2012"]
    )

    image_path = None
    xml_path = None

    for split in candidate_splits:

        if split is None:
            continue

        img_candidate = (
            devkit / split / "JPEGImages" / f"{stem}.jpg"
        )

        xml_candidate = (
            devkit / split / "Annotations" / f"{stem}.xml"
        )

        if img_candidate.exists():

            image_path = img_candidate
            xml_path = xml_candidate

            break

    if image_path is None:
        return None

    try:

        image = Image.open(image_path).convert("RGB")

    except OSError:

        return None

    if xml_path is not None and xml_path.exists():

        draw = ImageDraw.Draw(image)

        try:

            root = ET.parse(xml_path).getroot()

            for obj in root.findall("object"):

                name_node = obj.find("name")
                box_node = obj.find("bndbox")

                if name_node is None or box_node is None:
                    continue

                coords = [
                    box_node.find(key)
                    for key in ("xmin", "ymin", "xmax", "ymax")
                ]

                if any(node is None for node in coords):
                    continue

                class_name = normalize_label(name_node.text)

                box = [
                    float(node.text)
                    for node in coords
                ]

                color = (
                    "#22c55e"
                    if highlight_cls is None or class_name == highlight_cls
                    else "#94a3b8"
                )

                draw.rectangle(
                    box,
                    outline=color,
                    width=3
                )

                draw.text(
                    (box[0] + 4, max(0, box[1] - 14)),
                    f"GT: {class_name}",
                    fill=color
                )

        except (ET.ParseError, OSError, TypeError, ValueError):

            pass

    return image


# =============================================================================
# STREAMLIT PAGE SETUP AND SHARED CONTROLS
# =============================================================================

st.set_page_config(
    page_title="VIS4AI Phase I: Lecture Concepts Explorer (YOLO26n)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(
    "VIS4AI Phase I: Lecture Concepts & Exploratory Studio (YOLO26n)"
)

st.caption(
    "Five VIS4AI lecture-inspired views: "
    "(1) Multi-Channel Attribute Small Multiples, "
    "(2) 2D Projection Space Explorer, "
    "(3) Projection Paths & Time Curves, "
    "(4) Error-Similarity Matrix Seriation & Sample Drill-Down, "
    "and (5) Instance Flow / ConfusionFlow."
)

df_all = load_and_enrich_data()

if df_all.empty:

    st.error(
        f"No usable data found. Expected CSV: {CSV_PATH}. "
        "Run offline_test.py first to create voc_behavioral_results.csv."
    )

    st.stop()


st.sidebar.header("VIS4AI Lecture Modules")

lecture_tab = st.sidebar.radio(
    "Select VIS4AI Visualization Technique:",
    [
        "1. Multi-Channel Small Multiples (Penguin Paradigm)",
        "2. 2D Projection Space Explorer (PCA / t-SNE Dual-View)",
        "3. Projection Path Explorer & Time Curves (★ Perfect)",
        "4. Error-Similarity Matrix Seriation & Sample View",
        "5. Instance Flow / ConfusionFlow across Thresholds",
    ],
)

st.sidebar.markdown("---")

st.sidebar.subheader("Global Evaluation Gate")

conf_gate = st.sidebar.slider(
    "Confidence Threshold (T_conf)",
    0.25,
    0.90,
    0.25,
    step=0.05
)

iou_gate = st.sidebar.slider(
    "IoU Match Threshold (T_IoU)",
    0.15,
    0.90,
    0.50,
    step=0.05
)

df_eval = evaluate_dynamic_state(
    df_all,
    conf_gate,
    iou_gate
)

with st.sidebar.expander(
    "Data quality / geometry source",
    expanded=False
):

    if "Geometry_Source" in df_all.columns:

        st.write(
            df_all["Geometry_Source"].value_counts(dropna=False)
        )

    st.caption(
        "Geometry from VOC XML is exact. CSV-supplied geometry is retained. "
        "Synthetic fallback values are only illustrative and should not be "
        "interpreted as measured box geometry."
    )


# =============================================================================
# TAB 1 — MULTI-CHANNEL SMALL MULTIPLES
# =============================================================================

if lecture_tab == "1. Multi-Channel Small Multiples (Penguin Paradigm)":

    st.subheader(
        "1. Multi-Channel Scatter Plots & Small Multiples (Slides 6–9 & 13)"
    )

    with st.expander(
        "ℹ️ The Penguin Paradigm: dedicated visual channels",
        expanded=False
    ):

        st.markdown(
            """
            In the lecture's penguin example, each variable is assigned a visual
            channel. Here, **X-position** represents detector confidence and
            **Y-position** represents an interpretable object/detection attribute
            such as relative box area, aspect ratio, distance from the image centre,
            scene crowding or IoU.

            Small-multiple panels separate super-categories or VOC classes.
            Color and marker shape encode additional attributes.

            The trend line and per-panel correlation help reveal whether object
            size, shape, position or scene crowding is associated with confidence.

            Correlation is descriptive and does not establish causality.
            """
        )

    c1, c2, c3, c4 = st.columns([1.5, 1.4, 1.0, 1.0])

    with c1:

        y_attr_label = st.selectbox(
            "Y-position channel (interpretable attribute):",
            [
                "Rel_Area_Pct (Object Size: % of Image Area)",
                "Aspect_Ratio (Box Width / Height Ratio)",
                "Center_Dist_Norm (Distance from Image Center)",
                "Scene_GT_Objects (Ground-Truth Objects in Image)",
                "IoU (Localization Overlap with Ground Truth)",
            ],
        )

        y_col = y_attr_label.split(" ")[0]

    with c2:

        facet_mode = st.selectbox(
            "Small-multiple panels:",
            [
                "By Super-Category (Animals & Person | Vehicles | Indoor Objects)",
                "Top 6 Most Frequent VOC Classes",
                "Custom Class Selection (Choose below)",
            ],
        )

    with c3:

        color_channel = st.selectbox(
            "Color channel:",
            [
                "Dynamic_Outcome",
                "Environmental_Condition",
                "Split",
            ],
        )

    with c4:

        shape_channel = st.selectbox(
            "Marker shape:",
            [
                "Environmental_Condition",
                "Split",
                "Super_Category",
            ],
        )

    plot_sub = df_eval[
        df_eval["Candidate_Confidence"].notna()
        & (~df_eval["Is_Duplicate_Candidate_Row"])
        & (df_eval["Dynamic_Outcome"] != "Filtered FP")
    ].copy()

    plot_sub = plot_sub.replace(
        [np.inf, -np.inf],
        np.nan
    )

    plot_sub = plot_sub.dropna(
        subset=["Candidate_Confidence", y_col]
    )

    if "Super-Category" in facet_mode:

        panel_categories = [
            "Animals & Person",
            "Vehicles",
            "Indoor Objects",
        ]

        plot_sub = plot_sub[
            plot_sub["Super_Category"].isin(panel_categories)
        ]

        facet_col = "Super_Category"

    elif "Top 6" in facet_mode:

        top_classes = (
            plot_sub["Target_Class"]
            .value_counts()
            .head(6)
            .index.tolist()
        )

        plot_sub = plot_sub[
            plot_sub["Target_Class"].isin(top_classes)
        ]

        facet_col = "Target_Class"

    else:

        chosen_classes = st.multiselect(
            "Select VOC classes for panels:",
            options=VOC_CLASSES_ALPHABETICAL,
            default=[
                c for c in [
                    "person", "dog", "cat",
                    "car", "chair", "sofa"
                ]
                if c in VOC_CLASSES_ALPHABETICAL
            ],
        )

        plot_sub = plot_sub[
            plot_sub["Target_Class"].isin(chosen_classes)
        ]

        facet_col = "Target_Class"

    if plot_sub.empty:

        st.warning(
            "No rows match the current filters and selected channels."
        )

    else:

        color_map = (
            OUTCOME_COLORS_5
            if color_channel == "Dynamic_Outcome"
            else None
        )

        try:

            fig_sm = px.scatter(
                plot_sub,
                x="Candidate_Confidence",
                y=y_col,
                facet_col=facet_col,
                facet_col_wrap=3,
                color=color_channel,
                symbol=shape_channel,
                color_discrete_map=color_map,
                trendline="ols",
                hover_data=[
                    "Sample_ID",
                    "Ground_Truth",
                    "Effective_Pred_Class",
                    "IoU",
                    "Rel_Area_Pct",
                    "Geometry_Source",
                ],
                opacity=0.82,
                height=(
                    650
                    if plot_sub[facet_col].nunique() > 3
                    else 480
                ),
            )

        except (ImportError, ModuleNotFoundError):

            st.info(
                "statsmodels is unavailable; "
                "showing scatter points without OLS trend lines."
            )

            fig_sm = px.scatter(
                plot_sub,
                x="Candidate_Confidence",
                y=y_col,
                facet_col=facet_col,
                facet_col_wrap=3,
                color=color_channel,
                symbol=shape_channel,
                color_discrete_map=color_map,
                hover_data=[
                    "Sample_ID",
                    "Ground_Truth",
                    "Effective_Pred_Class",
                    "IoU",
                ],
                opacity=0.82,
                height=(
                    650
                    if plot_sub[facet_col].nunique() > 3
                    else 480
                ),
            )

        fig_sm.update_traces(
            marker=dict(
                size=8,
                line=dict(width=0.5, color="white")
            ),
            selector=dict(mode="markers"),
        )

        fig_sm.update_layout(
            title=f"Confidence vs. {y_col} across {facet_col}",
            xaxis_title="Candidate Confidence",
            yaxis_title=y_col,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.06,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=90, b=40, l=40, r=30),
        )

        st.plotly_chart(
            fig_sm,
            use_container_width=True
        )

        # Automated correlation summary.

        st.markdown(
            "#### Automated behavioral pattern finder"
        )

        corr_rows = []

        for group_name, group_df in plot_sub.groupby(
            facet_col,
            observed=True
        ):

            if (
                len(group_df) >= 5
                and group_df[y_col].std() > 1e-9
                and group_df["Candidate_Confidence"].std() > 1e-9
            ):

                corr_value = float(
                    np.corrcoef(
                        group_df["Candidate_Confidence"],
                        group_df[y_col]
                    )[0, 1]
                )

            else:

                corr_value = np.nan

            if pd.isna(corr_value):

                interpretation = (
                    "Too few observations or no variation "
                    "for a stable correlation."
                )

            elif corr_value >= 0.20:

                interpretation = (
                    f"Higher {y_col} is associated with higher "
                    f"confidence (r = {corr_value:+.2f})."
                )

            elif corr_value <= -0.15:

                interpretation = (
                    f"Higher {y_col} is associated with lower "
                    f"confidence (r = {corr_value:+.2f})."
                )

            else:

                interpretation = (
                    f"Confidence is weakly associated with "
                    f"{y_col} (r = {corr_value:+.2f})."
                )

            high_conf_errors = int(
                (
                    (group_df["Candidate_Confidence"] >= 0.75)
                    & (
                        group_df["Dynamic_Outcome"]
                        != "TP (Conf, IoU>T)"
                    )
                ).sum()
            )

            corr_rows.append(
                {
                    "Panel / Group": group_name,
                    "Instances": len(group_df),
                    "Mean Confidence": round(
                        float(group_df["Candidate_Confidence"].mean()),
                        3
                    ),
                    f"Mean {y_col}": round(
                        float(group_df[y_col].mean()),
                        3
                    ),
                    "Pearson r": (
                        round(corr_value, 3)
                        if pd.notna(corr_value)
                        else np.nan
                    ),
                    "High-Confidence Errors (≥ 0.75)": high_conf_errors,
                    "Interpretation": interpretation,
                }
            )

        st.dataframe(
            pd.DataFrame(corr_rows),
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# TAB 2 — 2D PROJECTION SPACE EXPLORER
# =============================================================================

elif lecture_tab == "2. 2D Projection Space Explorer (PCA / t-SNE Dual-View)":

    st.subheader(
        "2. 2D Projection Space Explorer: "
        "Dual-View Behavioral Projection (Slides 10–13)"
    )

    with st.expander(
        "ℹ️ Why project high-dimensional detection behavior into 2D?",
        expanded=False
    ):

        st.markdown(
            """
            Multiple geometric and behavioral attributes are standardized and
            projected into two dimensions.

            The left view colors the same points by class, super-category or
            outcome; the right view colors them by a continuous measure such
            as confidence or IoU.

            The neighborhood purity values summarize how often nearby points
            share a label.

            PCA is a linear projection. t-SNE emphasizes local neighborhoods and
            can change the global layout; distances between distant clusters in
            a t-SNE plot should therefore be interpreted cautiously.
            """
        )

    p_col1, p_col2, p_col3 = st.columns([1.2, 1.2, 1.2])

    with p_col1:

        proj_method = st.radio(
            "Dimensionality reduction:",
            [
                "PCA (Linear Principal Components)",
                "t-SNE (Non-Linear Neighborhood Manifold)",
            ],
        )

    with p_col2:

        left_color_by = st.selectbox(
            "Left view color encoding:",
            [
                "Super_Category",
                "Dynamic_Outcome",
                "Environmental_Condition",
                "Target_Class",
            ],
        )

    with p_col3:

        right_color_by = st.selectbox(
            "Right view continuous metric:",
            [
                "Candidate_Confidence (Model Confidence)",
                "IoU (Localization Overlap)",
                "Rel_Area_Pct (Relative Box Area %)",
            ],
        )

        right_col_name = right_color_by.split(" ")[0]

    proj_df = df_eval[
        df_eval["Candidate_Confidence"].notna()
        & (~df_eval["Is_Duplicate_Candidate_Row"])
    ].copy().reset_index(drop=True)

    if len(proj_df) < 3:

        st.warning(
            "At least three detection items with a candidate confidence "
            "are required for this projection."
        )

    else:

        class_tp_rate = proj_df.groupby(
            "Target_Class"
        )["Dynamic_Outcome"].transform(
            lambda values: (
                values == "TP (Conf, IoU>T)"
            ).mean()
        )

        proj_df["Class_TP_Rate"] = class_tp_rate

        # Construct behavioral and geometric feature matrix.

        feature_columns = [
            "Candidate_Confidence",
            "IoU",
            "Rel_Area_Pct",
            "Aspect_Ratio",
            "Center_X_Norm",
            "Center_Y_Norm",
            "Center_Dist_Norm",
            "Scene_GT_Objects",
            "Class_TP_Rate",
        ]

        numeric_features = proj_df[
            feature_columns
        ].apply(pd.to_numeric, errors="coerce")

        numeric_features = numeric_features.replace(
            [np.inf, -np.inf],
            np.nan
        )

        numeric_features["Rel_Area_Pct"] = np.log1p(
            numeric_features["Rel_Area_Pct"].clip(lower=0)
        )

        numeric_features["Aspect_Ratio"] = np.log1p(
            numeric_features["Aspect_Ratio"].clip(lower=0)
        )

        # One-hot encoding for nominal super-categories.

        category_dummies = pd.get_dummies(
            proj_df["Super_Category"].where(
                proj_df["Super_Category"].isin(
                    [
                        "Animals & Person",
                        "Vehicles",
                        "Indoor Objects",
                    ]
                ),
                "Background",
            ),
            prefix="SuperCategory",
            dtype=float,
        )

        numeric_features = pd.concat(
            [numeric_features, category_dummies],
            axis=1
        )

        numeric_features = (
            numeric_features
            .fillna(numeric_features.median())
            .fillna(0.0)
        )

        X_std = StandardScaler().fit_transform(
            numeric_features.to_numpy(dtype=float)
        )

        if "PCA" in proj_method:

            reducer = PCA(
                n_components=2,
                random_state=42
            )

            embedding = reducer.fit_transform(X_std)

            axis_x = (
                f"PC1 "
                f"({reducer.explained_variance_ratio_[0] * 100:.1f}% var)"
            )

            axis_y = (
                f"PC2 "
                f"({reducer.explained_variance_ratio_[1] * 100:.1f}% var)"
            )

            variance_label = (
                f"{reducer.explained_variance_ratio_.sum() * 100:.1f}%"
            )

        else:

            perplexity = min(
                30.0,
                max(2.0, (len(proj_df) - 1) / 3.0)
            )

            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                init="pca",
                learning_rate="auto",
            )

            embedding = reducer.fit_transform(X_std)

            axis_x = "t-SNE Dimension 1"
            axis_y = "t-SNE Dimension 2"

            variance_label = "not defined for t-SNE"

        proj_df["Dim1"] = embedding[:, 0]
        proj_df["Dim2"] = embedding[:, 1]

        # Compute neighborhood purity metrics.

        n_neighbors = min(11, len(proj_df))

        neighbors = NearestNeighbors(
            n_neighbors=n_neighbors
        ).fit(embedding)

        _, neighbor_indices = neighbors.kneighbors(embedding)

        if n_neighbors > 1:

            outcome_values = (
                proj_df["Dynamic_Outcome"].to_numpy()
            )

            category_values = (
                proj_df["Super_Category"].to_numpy()
            )

            outcome_purity = float(
                np.mean(
                    [
                        np.mean(
                            outcome_values[row[1:]]
                            == outcome_values[row[0]]
                        )
                        for row in neighbor_indices
                    ]
                )
            )

            category_purity = float(
                np.mean(
                    [
                        np.mean(
                            category_values[row[1:]]
                            == category_values[row[0]]
                        )
                        for row in neighbor_indices
                    ]
                )
            )

        else:

            outcome_purity = np.nan
            category_purity = np.nan

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Projected detection items",
            f"{len(proj_df):,}"
        )

        m2.metric(
            "KNN outcome purity (k=10)",
            f"{outcome_purity:.2f}"
            if pd.notna(outcome_purity)
            else "n/a"
        )

        m3.metric(
            "KNN super-category purity (k=10)",
            f"{category_purity:.2f}"
            if pd.notna(category_purity)
            else "n/a"
        )

        st.caption(
            f"Projection: {proj_method}. "
            f"Explained variance: {variance_label}."
        )

        v_left, v_right = st.columns(2)

        # LEFT VIEW

        with v_left:

            left_color_map = (
                OUTCOME_COLORS_5
                if left_color_by == "Dynamic_Outcome"
                else SUPER_COLORS
                if left_color_by == "Super_Category"
                else None
            )

            fig_left = px.scatter(
                proj_df,
                x="Dim1",
                y="Dim2",
                color=left_color_by,
                color_discrete_map=left_color_map,
                hover_data=[
                    "Sample_ID",
                    "Target_Class",
                    "Effective_Pred_Class",
                    "Candidate_Confidence",
                    "IoU",
                    "Rel_Area_Pct",
                ],
                opacity=0.82,
                height=540,
                title=f"A. Same projection colored by {left_color_by}",
            )

            fig_left.update_traces(
                marker=dict(
                    size=7,
                    line=dict(width=0.4, color="white")
                )
            )

            fig_left.update_layout(
                xaxis_title=axis_x,
                yaxis_title=axis_y,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
            )

            st.plotly_chart(
                fig_left,
                use_container_width=True
            )

        # RIGHT VIEW

        with v_right:

            fig_right = px.scatter(
                proj_df,
                x="Dim1",
                y="Dim2",
                color=right_col_name,
                color_continuous_scale="Viridis",
                hover_data=[
                    "Sample_ID",
                    "Target_Class",
                    "Dynamic_Outcome",
                    "Candidate_Confidence",
                    "IoU",
                    "Rel_Area_Pct",
                ],
                opacity=0.85,
                height=540,
                title=f"B. Same projection colored by {right_col_name}",
            )

            fig_right.update_traces(
                marker=dict(
                    size=7,
                    line=dict(width=0.4, color="white")
                )
            )

            fig_right.update_layout(
                xaxis_title=axis_x,
                yaxis_title=axis_y
            )

            st.plotly_chart(
                fig_right,
                use_container_width=True
            )


# =============================================================================
# TAB 3 — PROJECTION PATH EXPLORER AND TIME CURVES
# =============================================================================

elif lecture_tab == "3. Projection Path Explorer & Time Curves (★ Perfect)":

    st.subheader(
        "3. Projection Paths & Time Curves "
        "toward ★ Perfect Classification (Slide 14)"
    )

    with st.expander(
        "ℹ️ Time curves and projected decision paths",
        expanded=False
    ):

        st.markdown(
            """
            At every threshold step, each VOC class is represented by a
            behavioral state vector containing recall, precision, F1,
            mean confidence, mean IoU and error share.

            All states are projected together with an ideal reference vector.

            Connecting consecutive states creates one path per class.

            The star is an ideal reference point in this standardized PCA
            feature space; it is not a separately trained model.
            """
        )

    tc1, tc2 = st.columns([1.5, 1.5])

    with tc1:

        path_sweep_mode = st.radio(
            "Sequence parameter:",
            [
                f"Confidence threshold sequence "
                f"(0.25 → 0.90 at fixed IoU = {iou_gate:.2f})",

                f"IoU requirement sequence "
                f"(0.20 → 0.90 at fixed Conf = {conf_gate:.2f})",
            ],
        )

    with tc2:

        highlight_classes = st.multiselect(
            "Highlight class paths (empty = all classes):",
            options=VOC_CLASSES_ALPHABETICAL,
            default=[
                c for c in [
                    "aeroplane", "cat", "dog",
                    "person", "car", "chair", "sofa"
                ]
                if c in VOC_CLASSES_ALPHABETICAL
            ],
        )

    if "Confidence threshold" in path_sweep_mode:

        sequence_values = np.round(
            np.arange(0.25, 0.901, 0.05),
            2
        )

        threshold_pairs = [
            (float(value), float(iou_gate))
            for value in sequence_values
        ]

        step_labels = [
            f"Conf={value:.2f}"
            for value in sequence_values
        ]

    else:

        sequence_values = np.round(
            np.arange(0.20, 0.901, 0.05),
            2
        )

        threshold_pairs = [
            (float(conf_gate), float(value))
            for value in sequence_values
        ]

        step_labels = [
            f"IoU={value:.2f}"
            for value in sequence_values
        ]

    path_records = []

    base_for_sweep = df_all[
        ~df_all["Is_Duplicate_Candidate_Row"]
    ].copy()

    for step_index, (
        (conf_threshold, iou_threshold),
        step_label
    ) in enumerate(
        zip(threshold_pairs, step_labels)
    ):

        step_df = evaluate_dynamic_state(
            base_for_sweep,
            conf_threshold,
            iou_threshold
        )

        ground_truth_df = step_df[
            step_df["Ground_Truth"] != BACKGROUND_LABEL
        ]

        for class_name in VOC_CLASSES_ALPHABETICAL:

            class_gt = ground_truth_df[
                ground_truth_df["Ground_Truth"] == class_name
            ]

            gt_count = len(class_gt)

            true_positives = class_gt[
                (
                    class_gt["Dynamic_Outcome"]
                    == "TP (Conf, IoU>T)"
                )
                & (
                    class_gt["Effective_Pred_Class"]
                    == class_name
                )
            ]

            tp_count = len(true_positives)

            predicted_as_class = step_df[
                (
                    step_df["Effective_Pred_Class"]
                    == class_name
                )
                & step_df["Candidate_Confidence"].notna()
                & (
                    step_df["Candidate_Confidence"]
                    >= conf_threshold
                )
            ]

            fp_count = max(
                0,
                len(predicted_as_class) - tp_count
            )

            recall = (
                tp_count / gt_count * 100.0
                if gt_count
                else 0.0
            )

            precision = (
                tp_count / (tp_count + fp_count) * 100.0
                if (tp_count + fp_count) > 0
                else 0.0
            )

            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            mean_conf = (
                float(
                    true_positives["Candidate_Confidence"].mean()
                    * 100.0
                )
                if tp_count
                else 0.0
            )

            mean_iou = (
                float(true_positives["IoU"].mean() * 100.0)
                if tp_count
                else 0.0
            )

            error_share = (
                fp_count / (tp_count + fp_count) * 100.0
                if (tp_count + fp_count) > 0
                else 0.0
            )

            path_records.append(
                {
                    "Class": class_name,
                    "Super_Category": SUPER_CATEGORY_MAP[class_name],
                    "Step_Idx": step_index,
                    "Step_Label": step_label,
                    "Recall": round(recall, 2),
                    "Precision": round(precision, 2),
                    "F1": round(f1, 2),
                    "Mean_Conf": round(mean_conf, 2),
                    "Mean_IoU": round(mean_iou, 2),
                    "Error_Share": round(error_share, 2),
                    "TP": tp_count,
                    "FP": fp_count,
                    "GT": gt_count,
                }
            )

    path_df = pd.DataFrame(path_records)

    feature_columns = [
        "Recall",
        "Precision",
        "F1",
        "Mean_Conf",
        "Mean_IoU",
        "Error_Share",
    ]

    ideal_vector = np.array(
        [[100.0, 100.0, 100.0, 100.0, 100.0, 0.0]]
    )

    scaler_path = StandardScaler()

    standardized_states = scaler_path.fit_transform(
        np.vstack(
            [
                path_df[feature_columns].to_numpy(dtype=float),
                ideal_vector,
            ]
        )
    )

    path_pca = PCA(
        n_components=2,
        random_state=42
    )

    path_coordinates = path_pca.fit_transform(
        standardized_states
    )

    # Orient PC1 so the ideal reference appears on the right.

    if path_coordinates[-1, 0] < 0:

        path_coordinates[:, 0] *= -1.0

    path_df["PC1"] = path_coordinates[:-1, 0]
    path_df["PC2"] = path_coordinates[:-1, 1]

    star_x = float(path_coordinates[-1, 0])
    star_y = float(path_coordinates[-1, 1])

    fig_paths = go.Figure()

    for class_name in VOC_CLASSES_ALPHABETICAL:

        class_path = (
            path_df[path_df["Class"] == class_name]
            .sort_values("Step_Idx")
        )

        is_highlighted = (
            len(highlight_classes) == 0
            or class_name in highlight_classes
        )

        base_color = SUPER_COLORS[
            SUPER_CATEGORY_MAP[class_name]
        ]

        fig_paths.add_trace(
            go.Scatter(
                x=class_path["PC1"],
                y=class_path["PC2"],
                mode="lines+markers",
                name=f"{class_name} ({SUPER_CATEGORY_MAP[class_name]})",
                line=dict(
                    color=(
                        base_color
                        if is_highlighted
                        else "rgba(148, 163, 184, 0.20)"
                    ),
                    width=2.6 if is_highlighted else 1.0,
                    shape="spline",
                    smoothing=0.8,
                ),
                marker=dict(
                    size=(
                        np.linspace(5, 11, len(class_path))
                        if is_highlighted
                        else 4
                    ),
                    color=(
                        class_path["F1"]
                        if is_highlighted
                        else "rgba(148, 163, 184, 0.25)"
                    ),
                    colorscale="Viridis",
                    cmin=0,
                    cmax=100,
                    line=(
                        dict(width=0.8, color="#0f172a")
                        if is_highlighted
                        else dict(width=0)
                    ),
                ),
                customdata=np.stack(
                    [
                        class_path["Class"],
                        class_path["Step_Label"],
                        class_path["Precision"],
                        class_path["Recall"],
                        class_path["F1"],
                        class_path["Mean_IoU"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Class: <b>%{customdata[0]}</b><br>"
                    "Step: <b>%{customdata[1]}</b><br>"
                    "Precision: %{customdata[2]:.1f}% | "
                    "Recall: %{customdata[3]:.1f}%<br>"
                    "F1: <b>%{customdata[4]:.1f}%</b> | "
                    "Mean IoU: %{customdata[5]:.1f}%"
                    "<extra></extra>"
                ),
                showlegend=is_highlighted,
            )
        )

        if is_highlighted:

            fig_paths.add_annotation(
                x=float(class_path["PC1"].iloc[0]),
                y=float(class_path["PC2"].iloc[0]),
                text=f"<b>{class_name}</b>",
                showarrow=False,
                yshift=10,
                font=dict(
                    size=10,
                    color="#f8fafc"
                ),
                bgcolor="rgba(15, 23, 42, 0.75)",
            )

    # Perfect classification reference point.

    fig_paths.add_trace(
        go.Scatter(
            x=[star_x],
            y=[star_y],
            mode="markers+text",
            name="★ Perfect Classification",
            text=["<b>★ Perfect Classification</b>"],
            textposition="bottom left",
            textfont=dict(
                size=13,
                color="#facc15"
            ),
            marker=dict(
                symbol="star",
                size=24,
                color="#facc15",
                line=dict(
                    width=1.8,
                    color="#ffffff"
                )
            ),
        )
    )

    fig_paths.update_layout(
        height=650,
        title=(
            "Projected class paths in 2D PCA space — "
            f"{path_pca.explained_variance_ratio_.sum() * 100:.1f}% "
            "variance explained"
        ),
        xaxis_title=(
            "Behavioral Dimension 1 "
            f"({path_pca.explained_variance_ratio_[0] * 100:.1f}% var)"
        ),
        yaxis_title=(
            "Behavioral Dimension 2 "
            f"({path_pca.explained_variance_ratio_[1] * 100:.1f}% var)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )

    st.plotly_chart(
        fig_paths,
        use_container_width=True
    )

    st.caption(
        "The PCA star is the ideal reference vector after standardization. "
        "Path distance is a visual summary, not a calibrated performance score."
    )

    with st.expander(
        "Show threshold-sweep metrics",
        expanded=False
    ):

        st.dataframe(
            path_df,
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# TAB 4 — ERROR-SIMILARITY MATRIX SERIATION AND SAMPLE VIEW
# =============================================================================

elif lecture_tab == "4. Error-Similarity Matrix Seriation & Sample View":

    st.subheader(
        "4. Error-Similarity Matrix Seriation & "
        "Sample Drill-Down (Slides 15–20)"
    )

    with st.expander(
        "ℹ️ Error similarity, seriation and conceptual error types",
        expanded=False
    ):

        st.markdown(
            """
            The confusion matrix is reordered using hierarchical clustering of
            each class's off-diagonal error profile.

            Classes with similar error patterns are placed next to one another,
            which can reveal blocks that are difficult to see in alphabetical order.

            A sample view links a selected error to its source VOC image and
            ground-truth boxes.

            Confusions are additionally grouped into sibling classes, scene
            co-occurrence, cross-category visual ambiguity and taxonomy shifts.

            These labels are heuristic interpretations, not automatically
            verified semantic relations.
            """
        )

    s_col1, s_col2 = st.columns([1.3, 1.3])

    with s_col1:

        seriation_mode = st.radio(
            "Matrix ordering strategy:",
            [
                "Automatic Error-Similarity Seriation "
                "(Hierarchical Ward Clustering)",

                "Human Semantic Hierarchy "
                "(Animals / Vehicles / Indoor)",

                "Alphabetical (Unstructured Baseline)",
            ],
        )

    with s_col2:

        hide_diagonal = st.checkbox(
            "Zero out diagonal (focus on off-diagonal errors)",
            value=False,
            help=(
                "Hides correct-class counts to emphasize confusion "
                "between different classes."
            ),
        )

    # Build the active confusion matrix.

    matrix_rows = df_eval[
        df_eval["Ground_Truth"].isin(VOC_CLASSES_ALPHABETICAL)
        & df_eval["Effective_Pred_Class"].isin(VOC_CLASSES_ALPHABETICAL)
        & df_eval["Dynamic_Outcome"].isin(
            [
                "TP (Conf, IoU>T)",
                "Misclassification",
            ]
        )
        & (~df_eval["Is_Duplicate_Candidate_Row"])
    ].copy()

    raw_cm20 = pd.crosstab(
        matrix_rows["Ground_Truth"],
        matrix_rows["Effective_Pred_Class"]
    ).reindex(
        index=VOC_CLASSES_ALPHABETICAL,
        columns=VOC_CLASSES_ALPHABETICAL,
        fill_value=0
    )

    # Automatic error similarity seriation.

    if "Automatic" in seriation_mode:

        off_diagonal = raw_cm20.to_numpy(dtype=float).copy()

        np.fill_diagonal(
            off_diagonal,
            0.0
        )

        row_sums = off_diagonal.sum(
            axis=1,
            keepdims=True
        )

        row_normalized = np.divide(
            off_diagonal,
            row_sums,
            out=np.zeros_like(off_diagonal),
            where=row_sums > 0
        )

        # Include outgoing and incoming error profiles.

        similarity_profiles = np.hstack(
            [
                row_normalized,
                row_normalized.T,
            ]
        )

        if len(VOC_CLASSES_ALPHABETICAL) > 2:

            hierarchy = linkage(
                similarity_profiles,
                method="ward",
                optimal_ordering=True
            )

            leaf_order = leaves_list(hierarchy)

            ordered_classes = [
                VOC_CLASSES_ALPHABETICAL[index]
                for index in leaf_order
            ]

        else:

            ordered_classes = VOC_CLASSES_ALPHABETICAL

    elif "Human Semantic" in seriation_mode:

        category_order = {
            "Animals & Person": 0,
            "Vehicles": 1,
            "Indoor Objects": 2,
        }

        ordered_classes = sorted(
            VOC_CLASSES_ALPHABETICAL,
            key=lambda name: (
                category_order.get(
                    SUPER_CATEGORY_MAP[name],
                    9
                ),
                name
            )
        )

    else:

        ordered_classes = VOC_CLASSES_ALPHABETICAL

    displayed_cm = raw_cm20.reindex(
        index=ordered_classes,
        columns=ordered_classes
    ).copy()

    if hide_diagonal:

        np.fill_diagonal(
            displayed_cm.values,
            0
        )

    fig_matrix = px.imshow(
        displayed_cm,
        labels=dict(
            x="Predicted class",
            y="Ground-truth class",
            color="Instances"
        ),
        x=ordered_classes,
        y=ordered_classes,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=(
            "20 × 20 Confusion Matrix — "
            f"{seriation_mode.split('(')[0].strip()}"
        ),
    )

    fig_matrix.update_layout(
        height=750,
        margin=dict(
            l=10,
            r=10,
            t=70,
            b=100
        ),
    )

    fig_matrix.update_xaxes(
        tickangle=-45
    )

    st.plotly_chart(
        fig_matrix,
        use_container_width=True
    )

    # Conceptual confusion taxonomy.

    mis_rows = df_eval[
        (df_eval["Dynamic_Outcome"] == "Misclassification")
        & df_eval["Ground_Truth"].isin(VOC_CLASSES_ALPHABETICAL)
        & (~df_eval["Is_Duplicate_Candidate_Row"])
    ].copy()

    if mis_rows.empty:

        st.info(
            "No class misclassifications are present "
            "at the currently selected thresholds."
        )

    else:

        mis_rows["Conceptual_Relation"] = [
            classify_conceptual_confusion(gt, pred)
            for gt, pred in zip(
                mis_rows["Ground_Truth"],
                mis_rows["Effective_Pred_Class"]
            )
        ]

        pair_summary = (
            mis_rows.groupby(
                [
                    "Ground_Truth",
                    "Effective_Pred_Class",
                    "Conceptual_Relation",
                ],
                as_index=False,
            )
            .agg(
                Instances=("Record_ID", "count"),
                Mean_Confidence=("Candidate_Confidence", "mean"),
                Mean_IoU=("IoU", "mean"),
            )
            .sort_values(
                "Instances",
                ascending=False
            )
        )

        pair_summary["Mean_Confidence"] = (
            pair_summary["Mean_Confidence"].round(3)
        )

        pair_summary["Mean_IoU"] = (
            pair_summary["Mean_IoU"].round(3)
        )

        st.markdown(
            "#### Most frequent confusion pairs"
        )

        st.dataframe(
            pair_summary.head(15),
            use_container_width=True,
            hide_index=True
        )

        relation_counts = (
            mis_rows["Conceptual_Relation"]
            .value_counts()
            .rename_axis("Conceptual Relationship Type")
            .reset_index(name="Instances")
        )

        fig_relations = px.bar(
            relation_counts,
            x="Conceptual Relationship Type",
            y="Instances",
            color="Conceptual Relationship Type",
            title=(
                "Why does the detector confuse these classes? "
                "(heuristic taxonomy)"
            ),
        )

        fig_relations.update_layout(
            showlegend=False,
            height=350,
            xaxis_tickangle=-25
        )

        st.plotly_chart(
            fig_relations,
            use_container_width=True
        )

    # Sample drill-down.

    st.markdown("---")

    st.markdown(
        "### Sample View: inspect individual detections and failures"
    )

    drill_filter = st.selectbox(
        "Filter items to inspect:",
        [
            "All Semantic Confusions (GT ≠ Pred)",
            "Missed (IoU<T)",
            "Missed (Conf<T)",
            "Blind Spots (IoU=0)",
            "FP Hallucinations",
            "High-Confidence Errors (Conf ≥ 0.65)",
        ],
    )

    non_duplicate = df_eval[
        ~df_eval["Is_Duplicate_Candidate_Row"]
    ].copy()

    if drill_filter == "All Semantic Confusions (GT ≠ Pred)":

        inspection_df = non_duplicate[
            non_duplicate["Dynamic_Outcome"] == "Misclassification"
        ]

    elif drill_filter == "FP Hallucinations":

        inspection_df = non_duplicate[
            non_duplicate["Dynamic_Outcome"]
            == "FP (Hallucination)"
        ]

    elif drill_filter == "High-Confidence Errors (Conf ≥ 0.65)":

        inspection_df = non_duplicate[
            non_duplicate["Candidate_Confidence"].ge(0.65)
            & (
                non_duplicate["Dynamic_Outcome"]
                != "TP (Conf, IoU>T)"
            )
        ]

    else:

        inspection_df = non_duplicate[
            non_duplicate["Dynamic_Outcome"] == drill_filter
        ]

    if inspection_df.empty:

        st.info(
            "No items match this drill-down filter "
            "at the current thresholds."
        )

    else:

        inspection_df = inspection_df.copy()

        inspection_df["_row_label"] = inspection_df.apply(
            lambda row: (
                f"{row['Sample_ID']} | "
                f"GT: {row['Ground_Truth']} | "
                f"Pred: {row['Effective_Pred_Class']} | "
                f"{row['Dynamic_Outcome']}"
            ),
            axis=1
        )

        row_labels = inspection_df["_row_label"].tolist()

        selected_label = st.selectbox(
            "Select one detection item:",
            options=row_labels,
            index=0
        )

        selected_row = inspection_df.loc[
            inspection_df["_row_label"] == selected_label
        ].iloc[0]

        left_sample, right_sample = st.columns([1.1, 1.8])

        with left_sample:

            st.markdown(
                "#### Selected item"
            )

            st.write(
                f"**Sample:** {selected_row['Sample_ID']}"
            )

            st.write(
                f"**Ground truth:** {selected_row['Ground_Truth']}"
            )

            st.write(
                f"**Prediction:** {selected_row['Effective_Pred_Class']}"
            )

            st.write(
                f"**Outcome:** {selected_row['Dynamic_Outcome']}"
            )

            confidence_value = selected_row["Candidate_Confidence"]

            st.write(
                f"**Confidence:** {confidence_value:.3f}"
                if pd.notna(confidence_value)
                else "**Confidence:** unavailable"
            )

            st.write(
                f"**IoU:** {selected_row['IoU']:.3f}"
            )

            st.write(
                f"**Relative box area:** "
                f"{selected_row['Rel_Area_Pct']:.2f}%"
            )

            st.write(
                f"**Aspect ratio:** "
                f"{selected_row['Aspect_Ratio']:.2f}"
            )

            st.write(
                f"**Objects in scene:** "
                f"{int(selected_row['Scene_GT_Objects'])}"
            )

            st.write(
                f"**Geometry source:** "
                f"{selected_row['Geometry_Source']}"
            )

        with right_sample:

            sample_image = load_voc_sample_with_gt_boxes(
                str(selected_row["Sample_ID"]),
                highlight_cls=str(selected_row["Ground_Truth"])
            )

            if sample_image is None:

                st.warning(
                    "The original VOC image could not be found. "
                    "Check the VOCdevkit folder path."
                )

            else:

                st.image(
                    sample_image,
                    caption=(
                        f"{selected_row['Sample_ID']} — "
                        "green box: selected ground-truth class"
                    ),
                    use_container_width=True
                )


# =============================================================================
# TAB 5 — INSTANCE FLOW / CONFUSIONFLOW
# =============================================================================

elif lecture_tab == "5. Instance Flow / ConfusionFlow across Thresholds":

    st.subheader(
        "5. Instance Flow / ConfusionFlow across Thresholds (Slide 21)"
    )

    with st.expander(
        "ℹ️ How to read the instance flow",
        expanded=False
    ):

        st.markdown(
            """
            Each row represents one candidate / ground-truth record.

            Its outcome is evaluated at three successive threshold gates.

            The parallel categories diagram shows how records remain true
            positives, become misses or misclassifications, or are filtered
            as false positives as the gate becomes stricter.

            It is an instance-level view of the same records, not a tracking
            visualization across video frames.
            """
        )

    flow_type = st.radio(
        "Threshold sequence:",
        [
            "Confidence progression (vary confidence, fixed IoU)",
            "IoU progression (vary IoU, fixed confidence)",
        ],
    )

    if "Confidence progression" in flow_type:

        gate_values = [0.25, 0.50, 0.80]

        threshold_pairs = [
            (value, iou_gate)
            for value in gate_values
        ]

        gate_labels = [
            f"Gate {index + 1} (Conf ≥ {value:.2f})"
            for index, value in enumerate(gate_values)
        ]

    else:

        gate_values = [0.25, 0.50, 0.80]

        threshold_pairs = [
            (conf_gate, value)
            for value in gate_values
        ]

        gate_labels = [
            f"Gate {index + 1} (IoU ≥ {value:.2f})"
            for index, value in enumerate(gate_values)
        ]

    flow_base = df_all[
        ~df_all["Is_Duplicate_Candidate_Row"]
    ].copy().reset_index(drop=True)

    flow_outcomes = []

    for conf_threshold, iou_threshold in threshold_pairs:

        stage_df = evaluate_dynamic_state(
            flow_base,
            conf_threshold,
            iou_threshold
        )

        flow_outcomes.append(
            stage_df["Dynamic_Outcome"].to_numpy()
        )

    for gate_label, stage_outcomes in zip(
        gate_labels,
        flow_outcomes
    ):

        flow_base[gate_label] = stage_outcomes

    flow_color_values = {
        "TP (Conf, IoU>T)": 1.00,
        "Misclassification": 0.75,
        "FP (Hallucination)": 0.50,
        "Missed (IoU<T)": 0.30,
        "Missed (Conf<T)": 0.20,
        "Blind Spot (IoU=0)": 0.10,
        "Filtered FP": 0.00,
    }

    flow_base["Color_Code"] = (
        flow_base[gate_labels[-1]]
        .map(flow_color_values)
        .fillna(0.0)
    )

    dimensions = [
        {
            "label": gate_label,
            "values": flow_base[gate_label]
        }
        for gate_label in gate_labels
    ]

    fig_flow = px.parallel_categories(
        flow_base,
        dimensions=dimensions,
        color="Color_Code",
        color_continuous_scale="Viridis",
        labels={
            "Color_Code": "Final outcome score"
        },
        title=(
            "Instance-level outcome flow across "
            "three consecutive threshold gates"
        ),
        height=620,
    )

    fig_flow.update_layout(
        margin=dict(
            l=40,
            r=40,
            t=70,
            b=40
        )
    )

    st.plotly_chart(
        fig_flow,
        use_container_width=True
    )

    # Outcome counts at each threshold.

    st.markdown(
        "#### Outcome counts at each gate"
    )

    count_tables = []

    for gate_label in gate_labels:

        counts = flow_base[gate_label].value_counts()

        for outcome_name, count in counts.items():

            count_tables.append(
                {
                    "Threshold gate": gate_label,
                    "Outcome": outcome_name,
                    "Instances": int(count),
                }
            )

    st.dataframe(
        pd.DataFrame(count_tables),
        use_container_width=True,
        hide_index=True
    )

    # Inspect records that change outcome.

    st.markdown(
        "#### Inspect the records that change outcome"
    )

    flow_base["Changes across gates"] = (
        flow_base[gate_labels].nunique(axis=1) > 1
    )

    changed_only = st.checkbox(
        "Show only records whose outcome changes",
        value=True
    )

    inspect_flow = (
        flow_base[flow_base["Changes across gates"]]
        if changed_only
        else flow_base
    )

    st.dataframe(
        inspect_flow[
            [
                "Sample_ID",
                "Ground_Truth",
                "Effective_Pred_Class",
                "Candidate_Confidence",
                "IoU",
            ] + gate_labels
        ].head(200),
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

st.caption(
    "VIS4AI Phase I exploratory analysis. Results depend on the supplied "
    "behavioral CSV, the current confidence/IoU thresholds and the "
    "availability of VOC XML annotations. Synthetic fallback geometry "
    "is marked in the data-quality panel."
)