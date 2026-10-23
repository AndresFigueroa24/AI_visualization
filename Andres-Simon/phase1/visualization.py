from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import plotly.express as px
from ultralytics import YOLO

st.set_page_config(
    page_title="Phase I: Behavioral Analysis - YOLO26n on Pascal VOC",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# Pascal VOC 20 classes grouped semantically (Animals, Vehicles, Indoor, Outdoor items)
VOC_CLASSES_SEMANTIC = [
    # Animals
    "bird", "cat", "cow", "dog", "horse", "sheep", "person",
    # Vehicles
    "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
    # Indoor Objects
    "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
]

VOC_CLASSES_ALPHABETICAL = sorted(VOC_CLASSES_SEMANTIC)

@st.cache_resource
def load_model():
    model_path = BASE_DIR / "yolo26n.pt"
    if not model_path.exists():
        model_path = BASE_DIR / "yolov8n.pt"
    return YOLO(str(model_path))

@st.cache_data
def load_evaluation_data():
    csv_path = BASE_DIR / "voc_behavioral_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

model = load_model()
eval_df = load_evaluation_data()

st.title("YOLO26n Behavioral & Diagnostic Tool (Phase I)")
st.caption(
    "Black-Box Behavioral Inspection: Analyzing input-output patterns, scale sensitivity, "
    "and semantic class confusions across Pascal VOC."
)

st.sidebar.header("Navigation & Settings")
analysis_mode = st.sidebar.radio(
    "Select Inspection Mode",
    ["Interactive Single-Image Diagnostics", "Batch Failure Pattern Explorer"]
)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, step=0.05)
iou_threshold = st.sidebar.slider("IoU (NMS) Threshold", 0.1, 0.9, 0.45, step=0.05)

# ==============================================================================
# MODE 1: Interactive Single-Image Exploration with Perturbations
# ==============================================================================
if analysis_mode == "Interactive Single-Image Diagnostics":
    st.subheader("1. Hypothesis Testing & Perturbations")
    st.markdown(
        "Evaluate model stability under systematic input transformations (brightness, contrast, noise)."
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload image or pick from dataset", 
        type=["jpg", "jpeg", "png"]
    )

    voc_sample_dir = BASE_DIR / "VOC_dataset" / "VOCdevkit" / "VOC2007" / "JPEGImages"
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")
    elif voc_sample_dir.exists() and any(voc_sample_dir.glob("*.jpg")):
        sample_list = [p.name for p in list(voc_sample_dir.glob("*.jpg"))[:25]]
        selected_sample = st.sidebar.selectbox("Or choose a VOC2007 sample:", sample_list)
        raw_image = Image.open(voc_sample_dir / selected_sample).convert("RGB")
    else:
        raw_image = Image.new("RGB", (640, 480), color=(120, 130, 140))
        st.info("Upload an image or place images in VOC_dataset/VOCdevkit/VOC2007/JPEGImages.")

    st.sidebar.subheader("Input Perturbations")
    b_val = st.sidebar.slider("Brightness", 0.2, 2.0, 1.0, 0.1)
    c_val = st.sidebar.slider("Contrast", 0.2, 2.0, 1.0, 0.1)
    noise = st.sidebar.slider("Gaussian Noise", 0, 50, 0, 5)

    img_perturbed = ImageEnhance.Brightness(raw_image).enhance(b_val)
    img_perturbed = ImageEnhance.Contrast(img_perturbed).enhance(c_val)

    if noise > 0:
        arr = np.array(img_perturbed).astype(np.int16)
        arr = np.clip(arr + np.random.normal(0, noise, arr.shape), 0, 255).astype(np.uint8)
        img_perturbed = Image.fromarray(arr)

    results = model.predict(img_perturbed, conf=conf_threshold, iou=iou_threshold, verbose=False)
    res = results[0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Perturbed Input (What goes in)**")
        st.image(img_perturbed, use_container_width=True)
    with c2:
        st.markdown("**Model Output (What comes out)**")
        st.image(res.plot()[:, :, ::-1], use_container_width=True)

    detections = []
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            cid = int(b.cls[0].item())
            detections.append({
                "Class": res.names.get(cid, str(cid)),
                "Confidence": round(float(b.conf[0].item()), 3),
                "BBox [x1, y1, x2, y2]": [round(x, 1) for x in b.xyxy[0].tolist()]
            })

    if detections:
        det_df = pd.DataFrame(detections)
        st.dataframe(det_df, use_container_width=True)
        fig = px.bar(
            det_df, x="Class", y="Confidence", color="Confidence",
            range_y=[0, 1], title="Detected Instances Confidence"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No objects detected at this threshold.")

# ==============================================================================
# MODE 2: Multi-Sample Pattern & Diagnostic Test Bench (With Confusion Matrix)
# ==============================================================================
else:
    st.subheader("2. Real Diagnostic Test Bench Explorer")

    if eval_df.empty:
        st.error(
            "No evaluation data found! Run `python offline_test.py` in your terminal first "
            "to generate `voc_behavioral_results.csv`."
        )
    else:
        st.markdown(f"Loaded **{len(eval_df)}** real evaluation records from Pascal VOC.")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            split_filter = st.multiselect(
                "Dataset Split:",
                eval_df["Split"].unique(),
                default=list(eval_df["Split"].unique())
            )
        with col_f2:
            scale_filter = st.multiselect(
                "Object Scale Condition:",
                eval_df["Environmental_Condition"].unique(),
                default=list(eval_df["Environmental_Condition"].unique())
            )

        filtered = eval_df[
            (eval_df["Split"].isin(split_filter)) & 
            (eval_df["Environmental_Condition"].isin(scale_filter))
        ]

        # ----------------------------------------------------------------------
        # Investigative Section: Confusion Matrix
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Confusion Matrix as an Investigative Tool")
        st.markdown(
            "Examine whether errors follow **conceptual boundaries** (e.g. animal-to-animal or vehicle-to-vehicle) "
            "or indicate arbitrary confusion."
        )

        cm_col1, cm_col2, cm_col3 = st.columns([1, 1, 1])
        with cm_col1:
            order_mode = st.radio(
                "Class Ordering:",
                ["Semantic Grouping (Animals / Vehicles / Indoor)", "Alphabetical"],
                help="Reordering reveals cluster blocks corresponding to super-categories."
            )
        with cm_col2:
            include_bg = st.checkbox(
                "Include Missed Targets & False Positives", 
                value=True,
                help="Shows background hallucinations and missed annotations."
            )
        with cm_col3:
            normalize_mode = st.radio(
                "Matrix Normalization:",
                ["Raw Counts", "Normalized by Ground Truth (Recall)"]
            )

        # Build class lists
        active_classes = VOC_CLASSES_SEMANTIC if "Semantic" in order_mode else VOC_CLASSES_ALPHABETICAL
        
        y_classes = list(active_classes)
        x_classes = list(active_classes)
        
        if include_bg:
            y_classes = y_classes + ["None (Background)"]
            x_classes = x_classes + ["None"]

        # Filter evaluation rows relevant to the matrix
        matrix_df = filtered[
            (filtered["Ground_Truth"].isin(y_classes)) & 
            (filtered["Predicted_Class"].isin(x_classes))
        ]

        # Compute crosstab / confusion matrix
        cm = pd.crosstab(
            matrix_df["Ground_Truth"],
            matrix_df["Predicted_Class"],
            dropna=False
        ).reindex(index=y_classes, columns=x_classes, fill_value=0)

        # Normalization
        if normalize_mode == "Normalized by Ground Truth (Recall)":
            row_sums = cm.sum(axis=1)
            cm_plot = cm.div(row_sums.replace(0, 1), axis=0).round(2)
            color_label = "Recall Proportion"
        else:
            cm_plot = cm
            color_label = "Instance Count"

        fig_cm = px.imshow(
            cm_plot,
            labels=dict(x="Predicted Label (What comes out)", y="Ground Truth (What goes in)", color=color_label),
            x=x_classes,
            y=y_classes,
            color_continuous_scale="Blues",
            text_auto=True,
            aspect="auto"
        )
        fig_cm.update_layout(
            title="Interactive Diagnostic Confusion Matrix",
            xaxis_tickangle=-45,
            height=700
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # ----------------------------------------------------------------------
        # Other Behavioral Metrics
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Performance & Confidence Dispersion")

        fig_scale = px.histogram(
            filtered,
            x="Environmental_Condition",
            color="Outcome",
            barmode="group",
            title="Model Outcomes Grouped by Target Scale",
            color_discrete_map={
                "TP": "#2ca02c",
                "Misclassification": "#d62728",
                "False Positive": "#ff7f0e",
                "Missed Target": "#7f7f7f"
            }
        )
        st.plotly_chart(fig_scale, use_container_width=True)

        fig_strip = px.strip(
            filtered,
            x="Ground_Truth",
            y="Confidence",
            color="Outcome",
            hover_data=["Sample_ID", "Predicted_Class", "IoU"],
            title="Detections Confidence by Category",
            stripmode="overlay"
        )
        st.plotly_chart(fig_strip, use_container_width=True)

        st.markdown("#### Isolated Failure Cases (Misclassifications & Missed Targets)")
        failures = filtered[filtered["Outcome"] != "TP"]
        st.dataframe(failures, use_container_width=True)