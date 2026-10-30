from __future__ import annotations

"""Compact VIS4AI Phase I dashboard for YOLO / PASCAL VOC.

Place this file next to voc_behavioral_results.csv and run:
    streamlit run additional_visualizations.py

Optional VOC annotations/images are searched in:
    VOC_dataset/VOCdevkit
    archive/VOC_dataset/VOCdevkit
    VOCdevkit
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "voc_behavioral_results.csv"

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Three broad semantic groups for a readable overview of all 20 VOC classes.
SUPER_CATEGORY_MAP = {
    "bird": "Animals & People", "cat": "Animals & People",
    "cow": "Animals & People", "dog": "Animals & People",
    "horse": "Animals & People", "person": "Animals & People",
    "sheep": "Animals & People",
    "aeroplane": "Vehicles", "bicycle": "Vehicles", "boat": "Vehicles",
    "bus": "Vehicles", "car": "Vehicles", "motorbike": "Vehicles",
    "train": "Vehicles",
    "bottle": "Everyday Objects", "chair": "Everyday Objects",
    "diningtable": "Everyday Objects", "pottedplant": "Everyday Objects",
    "sofa": "Everyday Objects", "tvmonitor": "Everyday Objects",
}

CATEGORY_ORDER = ["Animals & People", "Vehicles", "Everyday Objects"]
CATEGORY_COLORS = {
    "Animals & People": "#2878B5",
    "Vehicles": "#2A9D68",
    "Everyday Objects": "#E09F3E",
}

OUTCOME_COLORS = {
    "Correct detection": "#279B62",
    "Class error": "#D9534F",
    "Localization / confidence miss": "#E09F3E",
    "False positive": "#8064A2",
    "Filtered false positive": "#9AA3AD",
    "No candidate box": "#596579",
}

BACKGROUND = "None (Background)"
NAME_MAP = {
    "airplane": "aeroplane", "motorcycle": "motorbike", "couch": "sofa",
    "dining table": "diningtable", "potted plant": "pottedplant",
    "tv": "tvmonitor", "television": "tvmonitor",
}


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def normalize_label(value: object) -> str:
    name = str(value).strip().lower()
    if name in {"", "none", "null", "nan"}:
        return "None"
    if name in {"background", "none (background)", "none(background)"}:
        return BACKGROUND
    if name in NAME_MAP:
        return NAME_MAP[name]
    return name.replace(" ", "").replace("_", "").replace("-", "")


def locate_devkit() -> Path | None:
    for path in (
        BASE_DIR / "VOC_dataset" / "VOCdevkit",
        BASE_DIR / "archive" / "VOC_dataset" / "VOCdevkit",
        BASE_DIR / "VOCdevkit",
    ):
        if path.is_dir():
            return path
    return None


def read_voc_geometry(devkit: Path) -> tuple[dict, dict]:
    """Read actual box geometry and object counts from VOC XML files."""
    boxes_by_image: dict = {}
    counts_by_image: dict = {}

    for split in ("VOC2007", "VOC2012"):
        annotation_dir = devkit / split / "Annotations"
        if not annotation_dir.exists():
            continue
        for xml_path in annotation_dir.glob("*.xml"):
            image_id = f"{split}_{xml_path.stem}"
            try:
                root = ET.parse(xml_path).getroot()
                size = root.find("size")
                if size is None:
                    continue
                width = float(size.findtext("width", "0"))
                height = float(size.findtext("height", "0"))
                if width <= 0 or height <= 0:
                    continue

                image_boxes: dict[str, list] = {}
                count = 0
                for obj in root.findall("object"):
                    class_name = normalize_label(obj.findtext("name", "None"))
                    box = obj.find("bndbox")
                    if box is None:
                        continue
                    x1 = float(box.findtext("xmin", "0"))
                    y1 = float(box.findtext("ymin", "0"))
                    x2 = float(box.findtext("xmax", "0"))
                    y2 = float(box.findtext("ymax", "0"))
                    box_w = max(1.0, x2 - x1)
                    box_h = max(1.0, y2 - y1)
                    area_pct = 100.0 * box_w * box_h / (width * height)
                    aspect = box_w / box_h
                    cx = (x1 + x2) / (2.0 * width)
                    cy = (y1 + y2) / (2.0 * height)
                    image_boxes.setdefault(class_name, []).append(
                        (area_pct, aspect, cx, cy)
                    )
                    count += 1
                boxes_by_image[image_id] = image_boxes
                counts_by_image[image_id] = count
            except (ET.ParseError, OSError, TypeError, ValueError, AttributeError):
                continue

    return boxes_by_image, counts_by_image


def split_sample_id(sample_id: object) -> tuple[str | None, str]:
    value = str(sample_id).strip()
    for split in ("VOC2007", "VOC2012"):
        prefix = split + "_"
        if value.startswith(prefix):
            return split, value[len(prefix):]
    return None, value


@st.cache_data(show_spinner="Loading VOC detection results…")
def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        return df

    defaults = {
        "Sample_ID": "unknown", "Outcome": "Unknown",
        "Predicted_Class": "None", "Ground_Truth": BACKGROUND,
        "Confidence": np.nan, "IoU": 0.0,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    df["Sample_ID"] = df["Sample_ID"].fillna("unknown").astype(str)
    df["Outcome"] = df["Outcome"].fillna("Unknown").astype(str)
    df["Predicted_Class"] = df["Predicted_Class"].fillna("None").map(normalize_label)
    df["Ground_Truth"] = df["Ground_Truth"].fillna(BACKGROUND).map(normalize_label)
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    df["IoU"] = pd.to_numeric(df["IoU"], errors="coerce").fillna(0).clip(0, 1)

    if "Record_ID" not in df.columns:
        df["Record_ID"] = [f"row_{i}" for i in range(len(df))]

    if "Candidate_Class" not in df.columns:
        df["Candidate_Class"] = np.where(
            df["Outcome"].eq("Missed Target"), "None", df["Predicted_Class"]
        )
    df["Candidate_Class"] = df["Candidate_Class"].fillna("None").map(normalize_label)

    if "Candidate_Confidence" not in df.columns:
        df["Candidate_Confidence"] = df["Confidence"]
    df["Candidate_Confidence"] = pd.to_numeric(
        df["Candidate_Confidence"], errors="coerce"
    )

    if "Is_Duplicate_Candidate_Row" not in df.columns:
        df["Is_Duplicate_Candidate_Row"] = False
    duplicate = df["Is_Duplicate_Candidate_Row"]
    if duplicate.dtype == object:
        df["Is_Duplicate_Candidate_Row"] = (
            duplicate.fillna("false").astype(str).str.lower().str.strip()
            .isin(["true", "1", "yes", "y"])
        )
    else:
        df["Is_Duplicate_Candidate_Row"] = duplicate.fillna(False).astype(bool)

    # Use geometry already present in the CSV, then fill missing GT geometry
    # from the exact VOC XML box. No synthetic geometry is generated.
    geometry_cols = ["Rel_Area_Pct", "Aspect_Ratio", "Center_X_Norm", "Center_Y_Norm"]
    for column in geometry_cols:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    devkit = locate_devkit()
    boxes_by_image, xml_counts = read_voc_geometry(devkit) if devkit else ({}, {})
    used: dict[tuple[str, str], int] = {}
    sources = []

    for idx, row in df.iterrows():
        image_id = str(row["Sample_ID"])
        gt_class = str(row["Ground_Truth"])
        values = None

        # A false positive has no corresponding GT object box.
        if gt_class != BACKGROUND:
            key = (image_id, gt_class)
            candidates = boxes_by_image.get(image_id, {}).get(gt_class, [])
            position = used.get(key, 0)
            if position < len(candidates):
                values = candidates[position]
                used[key] = position + 1

        if values is not None:
            csv_complete = all(pd.notna(row[col]) for col in geometry_cols)
            csv_partial = any(pd.notna(row[col]) for col in geometry_cols)
            for column, value in zip(geometry_cols, values):
                if pd.isna(df.at[idx, column]):
                    df.at[idx, column] = value
            if csv_complete:
                sources.append("CSV")
            elif csv_partial:
                sources.append("CSV + VOC XML")
            else:
                sources.append("VOC XML")
        elif all(pd.notna(row[col]) for col in geometry_cols):
            sources.append("CSV")
        else:
            sources.append("Unavailable")

    df["Geometry_Source"] = sources
    df["Center_Dist_Norm"] = np.sqrt(
        (df["Center_X_Norm"] - 0.5) ** 2
        + (df["Center_Y_Norm"] - 0.5) ** 2
    ) / np.sqrt(0.5)
    df["Center_Dist_Norm"] = df["Center_Dist_Norm"].clip(0, 1)

    # Category is determined by GT class (or predicted class for background rows).
    df["Target_Class"] = np.where(
        df["Ground_Truth"].ne(BACKGROUND),
        df["Ground_Truth"],
        df["Predicted_Class"],
    )
    df["Super_Category"] = df["Target_Class"].map(SUPER_CATEGORY_MAP)

    # XML gives exact scene object counts. CSV row counts are only a fallback proxy.
    fallback_counts = (
        df[df["Ground_Truth"].ne(BACKGROUND)]
        .groupby("Sample_ID")["Record_ID"].count().to_dict()
    )
    df["Scene_GT_Objects"] = df["Sample_ID"].map(
        lambda sid: xml_counts.get(sid, fallback_counts.get(sid, np.nan))
    )

    return df


def evaluate_outcomes(df: pd.DataFrame, conf_t: float, iou_t: float) -> pd.DataFrame:
    """Assign an outcome to every row at the chosen confidence/IoU gates."""
    out = df.copy()
    conf = pd.to_numeric(out["Candidate_Confidence"], errors="coerce").to_numpy(float)
    iou = pd.to_numeric(out["IoU"], errors="coerce").fillna(0).to_numpy(float)
    gt = out["Ground_Truth"].astype(str).to_numpy()
    original = out["Outcome"].astype(str).to_numpy()
    pred_class = out["Candidate_Class"].fillna("None").astype(str).to_numpy()
    fallback_class = out["Predicted_Class"].fillna("None").astype(str).to_numpy()
    pred_class = np.where(pred_class != "None", pred_class, fallback_class)

    is_gt = (gt != BACKGROUND) & (original != "False Positive")
    is_fp = ~is_gt
    has_candidate = np.isfinite(conf) & (conf >= 0)
    passes_conf = has_candidate & (conf >= conf_t)
    passes_iou = iou >= iou_t
    correct_class = pred_class == gt

    outcome = np.full(len(out), "No candidate box", dtype=object)
    outcome[is_gt & has_candidate & ~passes_conf & (iou > 0)] = "Localization / confidence miss"
    outcome[is_gt & passes_conf & ~passes_iou] = "Localization / confidence miss"
    outcome[is_gt & passes_conf & passes_iou & ~correct_class] = "Class error"
    outcome[is_gt & passes_conf & passes_iou & correct_class] = "Correct detection"
    outcome[is_fp & passes_conf] = "False positive"
    outcome[is_fp & ~passes_conf] = "Filtered false positive"

    out["Dynamic_Outcome"] = outcome
    out["Effective_Pred_Class"] = pred_class
    return out


def make_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build a numeric, standardized behavioral feature matrix for PCA/t-SNE."""
    feature_candidates = [
        "Candidate_Confidence", "IoU", "Rel_Area_Pct", "Aspect_Ratio",
        "Center_Dist_Norm",
    ]
    available = []
    for col in feature_candidates:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        # Keep geometry/crowding features only if they are available for enough rows.
        min_fraction = 0.70 if col in {"Rel_Area_Pct", "Aspect_Ratio", "Center_Dist_Norm"} else 0.0
        if values.notna().mean() >= min_fraction and values.nunique(dropna=True) > 1:
            available.append(col)

    features = df[available].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if "Rel_Area_Pct" in features:
        features["Rel_Area_Pct"] = np.log1p(features["Rel_Area_Pct"].clip(lower=0))
    if "Aspect_Ratio" in features:
        features["Aspect_Ratio"] = np.log1p(features["Aspect_Ratio"].clip(lower=0))

    features = features.fillna(features.median()).fillna(0.0)
    return features, available


def simplify_outcome(values: pd.Series) -> pd.Series:
    mapping = {
        "Correct detection": "Correct detection",
        "Class error": "Class error",
        "Localization / confidence miss": "Localization / confidence miss",
        "False positive": "False positive",
        "Filtered false positive": "Filtered false positive",
        "No candidate box": "No candidate box",
    }
    return values.map(mapping).fillna("No candidate box")


def simplify_flow_outcome(values: pd.Series) -> pd.Series:
    mapping = {
        "Correct detection": "Correct detection",
        "Class error": "Detection error",
        "Localization / confidence miss": "Detection error",
        "False positive": "False positive",
        "Filtered false positive": "Filtered false positive",
        "No candidate box": "Missed target",
    }
    return values.map(mapping).fillna("Missed target")


# -----------------------------------------------------------------------------
# Streamlit layout
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="VIS4AI Phase I — YOLO behavior",
    page_icon="📊",
    layout="wide",
)
st.title("VIS4AI Phase I — How does YOLO behave?")
st.caption("Three focused views of detection behavior on PASCAL VOC.")

raw = load_data()
if raw.empty:
    st.error(f"Could not find usable data. Expected: {CSV_PATH}")
    st.stop()

with st.sidebar:
    st.header("Evaluation settings")
    conf_gate = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
    iou_gate = st.slider("IoU threshold", 0.10, 0.90, 0.50, 0.05)
    st.caption("These thresholds update the outcome labels and the instance-flow view.")

# Exclude duplicate candidate rows from the visualizations.
df = raw.loc[~raw["Is_Duplicate_Candidate_Row"]].copy().reset_index(drop=True)
df = evaluate_outcomes(df, conf_gate, iou_gate)
df["Plot_Outcome"] = simplify_outcome(df["Dynamic_Outcome"])

view = st.sidebar.radio(
    "Visualization",
    [
        "1 · Object size vs. confidence",
        "2 · PCA and t-SNE feature spaces",
        "3 · Instance flow across thresholds",
    ],
)


# -----------------------------------------------------------------------------
# View 1: compact small multiples
# -----------------------------------------------------------------------------

if view == "1 · Object size vs. confidence":
    st.subheader("Object size and detector confidence")

    plot_df = df[
        df["Super_Category"].isin(CATEGORY_ORDER)
        & df["Candidate_Confidence"].notna()
        & df["Rel_Area_Pct"].notna()
        & (df["Rel_Area_Pct"] > 0)
        & df["Geometry_Source"].isin(["VOC XML", "CSV", "CSV + VOC XML"])
    ].copy()

    if plot_df.empty:
        st.warning(
            "No measured box geometry is available. Check that the VOC XML annotations "
            "or geometry columns in the CSV are present. Synthetic geometry is not plotted."
        )
    else:
        plot_df["Plot_Outcome"] = simplify_outcome(plot_df["Dynamic_Outcome"])
        fig = px.scatter(
            plot_df,
            x="Rel_Area_Pct",
            y="Candidate_Confidence",
            facet_col="Super_Category",
            facet_col_wrap=3,
            color="Plot_Outcome",
            category_orders={
                "Super_Category": CATEGORY_ORDER,
                "Plot_Outcome": list(OUTCOME_COLORS),
            },
            color_discrete_map=OUTCOME_COLORS,
            hover_data={
                "Sample_ID": True,
                "Ground_Truth": True,
                "Effective_Pred_Class": True,
                "IoU": ":.2f",
                "Rel_Area_Pct": ":.2f",
                "Candidate_Confidence": ":.2f",
                "Geometry_Source": True,
            },
            opacity=0.78,
            render_mode="webgl",
            height=500,
        )
        fig.update_xaxes(
            type="log",
            title_text="Object area (% of image, log scale)",
            showgrid=True,
            gridcolor="#E8ECF0",
        )
        fig.update_yaxes(
            title_text="YOLO confidence",
            range=[0, 1.02],
            showgrid=True,
            gridcolor="#E8ECF0",
        )
        fig.update_traces(marker=dict(size=7, line=dict(width=0.4, color="white")))
        fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
            legend=dict(
                title="Detection outcome",
                orientation="h",
                yanchor="bottom",
                y=1.10,
                xanchor="left",
                x=0,
                font=dict(size=10),
            ),
            margin=dict(l=55, r=25, t=105, b=55),
            height=500,
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Showing {len(plot_df):,} records with measured box geometry.")

        st.markdown(
            "**What the plot shows.** Each dot is one VOC object/detection record. "
            "The x-axis is the annotated bounding-box area as a percentage of the image; "
            "it is logarithmic so small objects remain visible. The y-axis is YOLO's "
            "confidence from 0 to 1. Colors distinguish correct detections, class errors, "
            "localization/confidence misses and false positives. Each panel is one broad "
            "semantic group. This makes it easy to see whether small objects or particular "
            "object groups tend to receive lower confidence."
        )


# -----------------------------------------------------------------------------
# View 2: PCA and t-SNE side-by-side
# -----------------------------------------------------------------------------

elif view == "2 · PCA and t-SNE feature spaces":
    st.subheader("Where do detection cases sit in feature space?")

    st.markdown(
        "The 20 PASCAL VOC classes are grouped into three broad semantic categories: "
        "**Animals & People** (living subjects), **Vehicles** (means of transport), and "
        "**Everyday Objects** (bottles, furniture, plants and monitors). This is a "
        "transparent, hand-defined grouping for visual comparison—not a grouping learned "
        "by YOLO and not an official VOC hierarchy. It keeps the overview readable while "
        "covering all 20 classes."
    )

    emb_df = df[
        df["Super_Category"].isin(CATEGORY_ORDER)
        & df["Candidate_Confidence"].notna()
    ].copy().reset_index(drop=True)

    # A fixed, reproducible sample keeps t-SNE responsive on larger CSV files.
    if len(emb_df) > 3000:
        emb_df = emb_df.sample(n=3000, random_state=42).reset_index(drop=True)

    features, used_features = make_feature_matrix(emb_df)

    if len(emb_df) < 4 or features.shape[1] < 2:
        st.warning(
            "At least four usable detection records and two varying numeric features "
            "are required for PCA/t-SNE. Check that confidence, IoU and box geometry "
            "are present in the CSV/XML data."
        )
    else:
        standardized = StandardScaler().fit_transform(features.to_numpy(dtype=float))

        pca = PCA(n_components=2, random_state=42)
        pca_xy = pca.fit_transform(standardized)

        perplexity = min(30.0, max(1.0, (len(emb_df) - 1) / 3.0))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        tsne_xy = tsne.fit_transform(standardized)

        emb_df["PCA 1"] = pca_xy[:, 0]
        emb_df["PCA 2"] = pca_xy[:, 1]
        emb_df["t-SNE 1"] = tsne_xy[:, 0]
        emb_df["t-SNE 2"] = tsne_xy[:, 1]

        left, right = st.columns(2)
        with left:
            fig_pca = px.scatter(
                emb_df,
                x="PCA 1",
                y="PCA 2",
                color="Super_Category",
                category_orders={"Super_Category": CATEGORY_ORDER},
                color_discrete_map=CATEGORY_COLORS,
                hover_data=["Target_Class", "Dynamic_Outcome", "Candidate_Confidence", "IoU"],
                opacity=0.78,
                render_mode="webgl",
                title=(
                    "PCA — linear overview<br>"
                    f"<sup>PC1: {pca.explained_variance_ratio_[0] * 100:.1f}% · "
                    f"PC2: {pca.explained_variance_ratio_[1] * 100:.1f}% of variance</sup>"
                ),
                height=480,
            )
            fig_pca.update_traces(marker=dict(size=6, line=dict(width=0.3, color="white")))
            fig_pca.update_layout(
                template="plotly_white", font=dict(size=11),
                xaxis_title="PC1", yaxis_title="PC2",
                legend=dict(title="Semantic group", orientation="h", y=-0.22, x=0),
                margin=dict(l=45, r=15, t=85, b=80),
            )
            st.plotly_chart(fig_pca, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                "**PCA components.** PC1 and PC2 are not original measurements. "
                "Each is a weighted combination of the standardized input features. "
                "PC1 captures the largest possible share of variation in the data; "
                "PC2 captures the largest remaining share while being perpendicular "
                "to PC1. The percentages in the title show how much total variation "
                "these two axes retain. Nearby points have similar feature profiles."
            )

        with right:
            fig_tsne = px.scatter(
                emb_df,
                x="t-SNE 1",
                y="t-SNE 2",
                color="Super_Category",
                category_orders={"Super_Category": CATEGORY_ORDER},
                color_discrete_map=CATEGORY_COLORS,
                hover_data=["Target_Class", "Dynamic_Outcome", "Candidate_Confidence", "IoU"],
                opacity=0.78,
                render_mode="webgl",
                title="t-SNE — local neighborhood structure",
                height=480,
            )
            fig_tsne.update_traces(marker=dict(size=6, line=dict(width=0.3, color="white")))
            fig_tsne.update_layout(
                template="plotly_white", font=dict(size=11),
                xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
                showlegend=False,
                margin=dict(l=45, r=15, t=85, b=80),
            )
            st.plotly_chart(fig_tsne, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                "**t-SNE components.** The two axes have no direct physical meaning "
                "and no percentage of explained variance. t-SNE tries to place cases "
                "with similar feature profiles near one another, so local groups and "
                "neighborhoods are the main thing to inspect. The apparent distance "
                "between far-apart clusters is not reliably meaningful."
            )

        st.caption(
            "Both plots use the same detection records, the same standardized input "
            f"features ({', '.join(used_features)}), and the same category colors. "
            "Compare the broad structure in PCA with the finer local neighborhoods in t-SNE."
        )


# -----------------------------------------------------------------------------
# View 3: corrected instance flow / parallel categories
# -----------------------------------------------------------------------------

else:
    st.subheader("How do detections change as the confidence threshold rises?")
    st.markdown(
        "Each ribbon represents one detection record. The three columns show how its "
        "outcome changes as the confidence threshold increases from 0.25 to 0.50 to "
        "0.80, while the IoU requirement remains fixed at the sidebar setting."
    )

    flow_df = df.copy().reset_index(drop=True)
    confidence_steps = [0.25, 0.50, 0.80]
    flow_columns = ["Confidence ≥ 0.25", "Confidence ≥ 0.50", "Confidence ≥ 0.80"]

    for threshold, column in zip(confidence_steps, flow_columns):
        stage = evaluate_outcomes(flow_df, threshold, iou_gate)
        flow_df[column] = simplify_flow_outcome(stage["Dynamic_Outcome"]).to_numpy()

    # Plotly's Graph Objects Parcats API expects one values-array per dimension.
    # Every dimension and the line-color vector must have exactly len(flow_df) entries.
    dimension_values = [flow_df[column].astype(str).tolist() for column in flow_columns]
    row_count = len(flow_df)
    if not all(len(values) == row_count for values in dimension_values):
        st.error("The instance-flow dimensions do not have matching lengths.")
        st.stop()

    final_codes = {
        "Correct detection": 4,
        "Detection error": 3,
        "Missed target": 2,
        "False positive": 1,
        "Filtered false positive": 0,
    }
    line_colors = flow_df[flow_columns[-1]].map(final_codes).astype(float).to_numpy()
    if len(line_colors) != row_count:
        st.error("The instance-flow color vector does not match the number of records.")
        st.stop()

    flow_order = [
        "Correct detection", "Detection error", "Missed target",
        "False positive", "Filtered false positive",
    ]
    dimensions = [
        dict(
            label=column,
            values=values,
            categoryorder="array",
            categoryarray=flow_order,
        )
        for column, values in zip(flow_columns, dimension_values)
    ]

    fig_flow = go.Figure(
        data=[
            go.Parcats(
                dimensions=dimensions,
                line=dict(
                    color=line_colors,
                    cmin=0,
                    cmax=4,
                    colorscale=[
                        [0.00, "#9AA3AD"],
                        [0.25, "#8064A2"],
                        [0.50, "#E09F3E"],
                        [0.75, "#D9534F"],
                        [1.00, "#279B62"],
                    ],
                    showscale=False,
                ),
                arrangement="perpendicular",
                bundlecolors=True,
                labelfont=dict(size=13, color="#263238"),
                tickfont=dict(size=11, color="#263238"),
                counts=True,
            )
        ]
    )
    fig_flow.update_layout(
        template="plotly_white",
        height=550,
        margin=dict(l=35, r=35, t=35, b=30),
        font=dict(size=12),
    )
    st.plotly_chart(fig_flow, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "Ribbon colors at the final gate: green = correct detection; red = detection error; "
        "orange = missed target; purple = false positive; grey = filtered false positive."
    )
    st.markdown(
        "**How to read the flow.** The columns are the three confidence gates; "
        "the bands show how the same records move between outcome groups. "
        "A correct detection should remain in *Correct detection*. False positives "
        "may move into *Filtered false positive* as the threshold increases. "
        "A rise in missed targets at stricter thresholds shows the trade-off: "
        "removing weak detections can also discard real objects. Colors indicate "
        "the outcome at the final (0.80) gate."
    )

st.markdown("---")
st.caption(
    "VIS4AI Phase I · The visualizations use the supplied behavioral CSV. "
    "Object geometry is taken from VOC XML or existing CSV fields; missing geometry "
    "is left out rather than replaced with synthetic values."
)
