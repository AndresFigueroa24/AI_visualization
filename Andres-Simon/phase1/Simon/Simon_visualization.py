from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ultralytics import YOLO

st.set_page_config(
    page_title="YOLO26n Visual Analytics & Behavioral Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Standard Pascal VOC 20 classes grouped semantically
VOC_CLASSES_SEMANTIC = [
    # Animals & Person
    "bird", "cat", "cow", "dog", "horse", "sheep", "person",
    # Vehicles
    "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
    # Indoor Objects
    "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
]
VOC_CLASSES_ALPHABETICAL = sorted(VOC_CLASSES_SEMANTIC)

PHASE_STATE_COLORS = {
    "TP": "#2ca02c",
    "Recovered TP (0 < IoU < 0.5)": "#00b894",
    "Misclassification": "#d62728",
    "False Positive": "#ff7f0e",
    "Localization Error (IoU < T)": "#1f77b4",
    "Filtered out (Greyed Out)": "#cbd5e1",
    "Missed Target": "#7f7f7f"
}

SCALE_ORDER = [
    "Small Scale (<4%)",
    "Medium Scale (4-20%)",
    "Large Scale (>20%)"
]

SCALE_SYMBOLS = {
    "Small Scale (<4%)": "square",
    "Medium Scale (4-20%)": "circle",
    "Large Scale (>20%)": "x"
}

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
    cleaned = str(name).strip().lower()
    if cleaned in VOC_NAME_MAP:
        return VOC_NAME_MAP[cleaned]
    return cleaned.replace(" ", "").replace("_", "").replace("-", "")


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "yolo26n.pt"
    if not model_path.exists():
        model_path = BASE_DIR / "yolov8n.pt"
    return YOLO(str(model_path))


@st.cache_data
def load_evaluation_data() -> pd.DataFrame:
    csv_path = BASE_DIR / "voc_behavioral_results.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["Predicted_Class"] = df["Predicted_Class"].fillna("None").astype(str)
    df["Ground_Truth"] = df["Ground_Truth"].fillna("None (Background)").astype(str)

    if "Record_ID" not in df.columns:
        df["Record_ID"] = [f"rec_{i:05d}" for i in range(len(df))]

    if "Candidate_Class" not in df.columns:
        df["Candidate_Class"] = np.where(
            df["Outcome"] == "Missed Target",
            "None",
            df["Predicted_Class"]
        )
    else:
        df["Candidate_Class"] = df["Candidate_Class"].fillna("None").astype(str)

    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    df["IoU"] = pd.to_numeric(df["IoU"], errors="coerce").fillna(0.0)

    # Ensure Blind Spot misses (IoU == 0) never create an artificial bar at Confidence = 0.0
    blind_miss_mask = (df["Outcome"] == "Missed Target") & (df["IoU"] <= 0.0)
    df.loc[blind_miss_mask, "Confidence"] = np.nan

    if "Candidate_Confidence" not in df.columns:
        df["Candidate_Confidence"] = df["Confidence"]
    else:
        df["Candidate_Confidence"] = pd.to_numeric(df["Candidate_Confidence"], errors="coerce")
        df.loc[blind_miss_mask, "Candidate_Confidence"] = np.nan

    if "Is_Duplicate_Candidate_Row" not in df.columns:
        missed_cands = df[
            (df["Outcome"] == "Missed Target") & (df["IoU"] > 0.0) & (df["Candidate_Confidence"].notna())
        ][["Sample_ID", "Candidate_Class", "Candidate_Confidence", "IoU"]].drop_duplicates()

        missed_set = set(
            zip(
                missed_cands["Sample_ID"],
                missed_cands["Candidate_Class"],
                missed_cands["Candidate_Confidence"].round(3),
                missed_cands["IoU"].round(3)
            )
        )
        dup_flags = []
        for _, r in df.iterrows():
            if r["Outcome"] == "False Positive" and pd.notna(r["Confidence"]):
                key = (r["Sample_ID"], r["Predicted_Class"], round(float(r["Confidence"]), 3), round(float(r["IoU"]), 3))
                dup_flags.append(key in missed_set)
            else:
                dup_flags.append(False)
        df["Is_Duplicate_Candidate_Row"] = dup_flags

    return df


def evaluate_at_thresholds_vectorized(
    df_in: pd.DataFrame,
    conf_t: float,
    iou_t: float,
    strict_grey_iou: bool = False
) -> pd.DataFrame:
    """
    Fast vectorized evaluation at (conf_t, iou_t):
    - Detections with Confidence < conf_t are suppressed by the model -> 'Filtered out (Greyed Out)'
      (applies to TP, False Positive, AND Misclassification).
    - Detections with Confidence >= conf_t on a GT row:
      * If IoU >= iou_t: 'TP' (or 'Recovered TP (0 < IoU < 0.5)') if class matches, else 'Misclassification'.
      * If 0 < IoU < iou_t: 'Localization Error' (blue) in diagnostic mode, or 'Filtered out (Greyed Out)' in strict mode.
    """
    d = df_in.copy()
    conf = d["Candidate_Confidence"].fillna(-1.0).to_numpy()
    iou = d["IoU"].fillna(0.0).to_numpy()
    orig = d["Outcome"].to_numpy()
    gt_cls = d["Ground_Truth"].to_numpy()
    cand_cls = d["Candidate_Class"].to_numpy()
    pred_cls = d["Predicted_Class"].to_numpy()

    eff_cls = np.where(cand_cls != "None", cand_cls, pred_cls)
    is_gt = orig != "False Positive"
    has_box = conf >= 0.0
    pass_conf = conf >= conf_t
    pass_iou = iou >= iou_t
    valid_cls = eff_cls != "None"
    same_cls = eff_cls == gt_cls

    dyn_out = np.full(len(d), "Missed Target", dtype=object)
    gt_active = is_gt & has_box & pass_conf
    dyn_out[gt_active & pass_iou & valid_cls & same_cls] = "TP"
    dyn_out[gt_active & pass_iou & valid_cls & (~same_cls)] = "Misclassification"
    dyn_out[gt_active & (~(pass_iou & valid_cls))] = "Localization Error"

    fp_rows = ~is_gt
    dyn_out[fp_rows & (~pass_conf)] = "Filtered FP"
    dyn_out[fp_rows & pass_conf] = "False Positive"

    p_state = np.full(len(d), "Filtered out (Greyed Out)", dtype=object)
    p_state[is_gt & (~has_box)] = "Blind Spot (No Box)"
    p_state[gt_active & pass_iou & valid_cls & same_cls & (orig != "Missed Target")] = "TP"
    p_state[gt_active & pass_iou & valid_cls & same_cls & (orig == "Missed Target")] = "Recovered TP (0 < IoU < 0.5)"
    p_state[gt_active & pass_iou & valid_cls & (~same_cls)] = "Misclassification"

    loc_err_mask = gt_active & (~(pass_iou & valid_cls))
    p_state[loc_err_mask] = "Filtered out (Greyed Out)" if strict_grey_iou else "Localization Error (IoU < T)"

    if strict_grey_iou:
        p_state[fp_rows & pass_conf & pass_iou] = "False Positive"
        p_state[fp_rows & pass_conf & (~pass_iou)] = "Filtered out (Greyed Out)"
    else:
        p_state[fp_rows & pass_conf] = "False Positive"

    d["Dynamic_Outcome"] = dyn_out
    d["Phase_State"] = p_state
    d["Effective_Pred_Class"] = eff_cls
    return d


def compute_all_steps_class_metrics(df_in: pd.DataFrame, threshold_pairs: list) -> dict:
    """
    Vectorized computation of per-class and global Precision, Recall, F1, and all underlying
    formula components (TP, Pure FP, Localization Error, Misclassification, Conf-Dropped FN, Blind Spot FN)
    across all threshold steps in < 50 ms.
    """
    orig = df_in["Outcome"].to_numpy()
    gt_cls = df_in["Ground_Truth"].to_numpy()
    cand_cls = df_in["Candidate_Class"].to_numpy()
    pred_cls = df_in["Predicted_Class"].to_numpy()
    eff_cls = np.where(cand_cls != "None", cand_cls, pred_cls)
    conf = df_in["Candidate_Confidence"].fillna(-1.0).to_numpy()
    iou = df_in["IoU"].fillna(0.0).to_numpy()
    is_dup = df_in["Is_Duplicate_Candidate_Row"].to_numpy()

    is_gt = orig != "False Positive"
    is_pure_fp_row = (orig == "False Positive") & (~is_dup)
    has_box = conf >= 0.0

    cls_list = VOC_CLASSES_ALPHABETICAL
    n_cls = len(cls_list)
    n_steps = len(threshold_pairs)

    gt_cls_masks = np.array([is_gt & (gt_cls == c) for c in cls_list])
    eff_cls_gt_masks = np.array([is_gt & (eff_cls == c) for c in cls_list])
    fp_cls_masks = np.array([is_pure_fp_row & (pred_cls == c) for c in cls_list])

    gt_counts = gt_cls_masks.sum(axis=1)
    blind_counts = np.array([(gt_cls_masks[i] & (~has_box)).sum() for i in range(n_cls)])

    rec_mat = np.zeros((n_steps, n_cls))
    prec_mat = np.zeros((n_steps, n_cls))
    f1_mat = np.zeros((n_steps, n_cls))
    err_share_mat = np.zeros((n_steps, n_cls))
    miss_share_mat = np.zeros((n_steps, n_cls))

    tp_mat = np.zeros((n_steps, n_cls), dtype=int)
    pure_fp_mat = np.zeros((n_steps, n_cls), dtype=int)
    loc_err_mat = np.zeros((n_steps, n_cls), dtype=int)
    mis_mat = np.zeros((n_steps, n_cls), dtype=int)
    conf_drop_fn_mat = np.zeros((n_steps, n_cls), dtype=int)
    blind_fn_mat = np.tile(blind_counts, (n_steps, 1))

    for s_idx, (c_t, i_t) in enumerate(threshold_pairs):
        pass_conf = conf >= c_t
        pass_iou = iou >= i_t
        valid_cls = eff_cls != "None"
        same_cls = eff_cls == gt_cls

        gt_active = is_gt & has_box & pass_conf
        tp_mask = gt_active & pass_iou & valid_cls & same_cls
        mis_mask = gt_active & pass_iou & valid_cls & (~same_cls)
        loc_mask = gt_active & (~(pass_iou & valid_cls))
        conf_drop_mask = is_gt & has_box & (~pass_conf)
        fp_active_mask = is_pure_fp_row & pass_conf

        tp_c = (gt_cls_masks & tp_mask).sum(axis=1)
        mis_c = (gt_cls_masks & mis_mask).sum(axis=1)
        loc_c = (eff_cls_gt_masks & loc_mask).sum(axis=1)
        conf_drop_c = (gt_cls_masks & conf_drop_mask).sum(axis=1)
        fp_c = (fp_cls_masks & fp_active_mask).sum(axis=1)

        total_fp_c = fp_c + loc_c + mis_c
        denom_p = tp_c + total_fp_c

        rec_c = np.where(gt_counts > 0, (tp_c / np.maximum(1, gt_counts)) * 100.0, 0.0)
        prec_c = np.where(denom_p > 0, (tp_c / np.maximum(1, denom_p)) * 100.0, np.where(tp_c == 0, 100.0, 0.0))
        f1_c = np.where((prec_c + rec_c) > 0, 2 * prec_c * rec_c / np.maximum(1e-6, prec_c + rec_c), 0.0)

        rec_mat[s_idx] = np.round(rec_c, 1)
        prec_mat[s_idx] = np.round(prec_c, 1)
        f1_mat[s_idx] = np.round(f1_c, 1)
        err_share_mat[s_idx] = np.round(100.0 - prec_c, 1)
        miss_share_mat[s_idx] = np.round(100.0 - rec_c, 1)

        tp_mat[s_idx] = tp_c
        pure_fp_mat[s_idx] = fp_c
        loc_err_mat[s_idx] = loc_c
        mis_mat[s_idx] = mis_c
        conf_drop_fn_mat[s_idx] = conf_drop_c

    return {
        "classes": cls_list,
        "gt_counts": gt_counts,
        "rec_mat": rec_mat,
        "prec_mat": prec_mat,
        "f1_mat": f1_mat,
        "err_share_mat": err_share_mat,
        "miss_share_mat": miss_share_mat,
        "tp_mat": tp_mat,
        "pure_fp_mat": pure_fp_mat,
        "loc_err_mat": loc_err_mat,
        "mis_mat": mis_mat,
        "conf_drop_fn_mat": conf_drop_fn_mat,
        "blind_fn_mat": blind_fn_mat,
        "total_fp_mat": pure_fp_mat + loc_err_mat + mis_mat,
        "total_fn_mat": conf_drop_fn_mat + loc_err_mat + mis_mat + blind_fn_mat,
    }


def build_animated_quadrant_and_formula_bars(
    df_in: pd.DataFrame,
    threshold_pairs: list,
    step_labels: list,
    color_mode: str,
    focus_target: str,
    show_counts_on_labels: bool
) -> go.Figure:
    """
    Animation 1:
    - Left Subplot: 2D Class Diagnostic Quadrant (Recall vs. Precision) with ZERO-JUMPING smooth trajectory lines
      (mode='lines' only, fixed index slice per class) and per-point color encoding.
    - Right Subplot: Live Animated Bar Chart of the exact Precision & Recall Formula Building Blocks
      (Active TP, Pure Background FP, Localization Error, Misclassification, Filtered by Conf, Blind Spot).
    - Bottom Live Formula HUD: Updates the mathematical equations for Precision and Recall with live numbers on every frame!
    """
    m = compute_all_steps_class_metrics(df_in, threshold_pairs)
    classes = m["classes"]
    n_cls = len(classes)
    n_steps = len(step_labels)
    gt_counts = m["gt_counts"]
    bubble_sizes = np.clip(np.sqrt(gt_counts) * 5.2, 13, 52)

    # Highlight ring if a single focus class is selected
    if focus_target in classes:
        focus_idx = classes.index(focus_target)
        line_widths = [3.2 if i == focus_idx else 1.1 for i in range(n_cls)]
        line_colors = ["#000000" if i == focus_idx else "#334155" for i in range(n_cls)]
    else:
        focus_idx = None
        line_widths = [1.3] * n_cls
        line_colors = ["#0f172a"] * n_cls

    # Select color matrix & colorbar configuration for the 20 class bubbles
    if color_mode == "Error Share in Predictions: FP_total / (TP + FP_total) [%]":
        c_mat = m["err_share_mat"]
        cscale = "RdYlGn_r"
        cmin, cmax = 0.0, 75.0
        cbar_title = "FP Share (%)"
    elif color_mode == "Active False Alarms Count (Pure FP + Loc.Err + Misclass)":
        c_mat = m["total_fp_mat"]
        cscale = "YlOrRd"
        cmin, cmax = 0.0, float(max(10, int(m["total_fp_mat"].max())))
        cbar_title = "Active FP Count"
    elif color_mode == "Missed Target Share: FN_total / GT_Count [%]":
        c_mat = m["miss_share_mat"]
        cscale = "YlOrBr"
        cmin, cmax = 0.0, 80.0
        cbar_title = "FN Share (%)"
    else:
        c_mat = m["f1_mat"]
        cscale = "Viridis"
        cmin, cmax = 20.0, 100.0
        cbar_title = "F1-Score (%)"

    # 1. Static full-sweep rail lines (subtle background curves so trajectories are clear from the start)
    rail_x, rail_y = [], []
    for c_i in range(n_cls):
        rail_x.extend(m["rec_mat"][:, c_i].tolist())
        rail_x.append(None)
        rail_y.extend(m["prec_mat"][:, c_i].tolist())
        rail_y.append(None)

    # 2. Fixed-length progressive trail builder (strictly n_cls * (n_steps + 1) elements on every frame -> ZERO jumping!)
    def make_fixed_length_trail(step_i: int):
        tx, ty = [], []
        for c_i in range(n_cls):
            r_hist = m["rec_mat"][:step_i + 1, c_i].tolist()
            p_hist = m["prec_mat"][:step_i + 1, c_i].tolist()
            # Pad remaining future slots with the current head position so slice indices never shift!
            pad_len = n_steps - len(r_hist)
            tx.extend(r_hist + [r_hist[-1]] * pad_len + [None])
            ty.extend(p_hist + [p_hist[-1]] * pad_len + [None])
        return tx, ty

    def make_bubble_labels(step_i: int):
        if not show_counts_on_labels:
            return classes
        return [
            f"{cls_name} [TP:{m['tp_mat'][step_i, idx]}|FP:{m['total_fp_mat'][step_i, idx]}|FN:{m['total_fn_mat'][step_i, idx]}]"
            for idx, cls_name in enumerate(classes)
        ]

    def get_bar_counts_and_formula_annotation(step_i: int, step_lbl: str):
        if focus_idx is not None:
            scope_title = f"Class '{focus_target.upper()}' (GT = {int(gt_counts[focus_idx])})"
            tp_v = int(m["tp_mat"][step_i, focus_idx])
            fp_bg_v = int(m["pure_fp_mat"][step_i, focus_idx])
            loc_v = int(m["loc_err_mat"][step_i, focus_idx])
            mis_v = int(m["mis_mat"][step_i, focus_idx])
            fn_conf_v = int(m["conf_drop_fn_mat"][step_i, focus_idx])
            fn_blind_v = int(m["blind_fn_mat"][step_i, focus_idx])
        else:
            scope_title = f"All 20 VOC Classes Combined (Total GT = {int(gt_counts.sum())})"
            tp_v = int(m["tp_mat"][step_i].sum())
            fp_bg_v = int(m["pure_fp_mat"][step_i].sum())
            loc_v = int(m["loc_err_mat"][step_i].sum())
            mis_v = int(m["mis_mat"][step_i].sum())
            fn_conf_v = int(m["conf_drop_fn_mat"][step_i].sum())
            fn_blind_v = int(m["blind_fn_mat"][step_i].sum())

        fp_tot = fp_bg_v + loc_v + mis_v
        fn_tot = fn_conf_v + loc_v + mis_v + fn_blind_v
        gt_tot = tp_v + fn_tot

        prec_pct = (tp_v / (tp_v + fp_tot) * 100.0) if (tp_v + fp_tot) > 0 else (100.0 if tp_v == 0 else 0.0)
        rec_pct = (tp_v / max(1, gt_tot) * 100.0)

        bar_vals = [tp_v, fp_bg_v, loc_v, mis_v, fn_conf_v, fn_blind_v]

        formula_html = (
            f"<b>Live Formula Breakdown at {step_lbl} — {scope_title}:</b><br>"
            f"• <b>Precision</b> = TP / (TP + FP<sub>BG</sub> + Loc.Err + Misclass) = "
            f"<b>{tp_v}</b> / (<b>{tp_v}</b> + {fp_bg_v} + {loc_v} + {mis_v}) = "
            f"<b>{tp_v} / {tp_v + fp_tot} = {prec_pct:.1f}%</b><br>"
            f"• <b>Recall</b> = TP / (TP + FN<sub>Conf&lt;T</sub> + Loc.Err + Misclass + BlindSpot) = "
            f"<b>{tp_v}</b> / (<b>{tp_v}</b> + {fn_conf_v} + {loc_v} + {mis_v} + {fn_blind_v}) = "
            f"<b>{tp_v} / {gt_tot} = {rec_pct:.1f}%</b>"
        )
        return bar_vals, formula_html

    bar_categories = [
        "Active TP\n( Drives P↑ & R↑ )",
        "Pure BG FP\n( Pulls P↓ )",
        "Loc. Error (IoU<T)\n( Pulls P↓ & R↓ )",
        "Misclassification\n( Pulls P↓ & R↓ )",
        "Missed (Conf<T)\n( Pulls R↓ )",
        "Blind Spot (IoU=0)\n( Pulls R↓ )"
    ]
    bar_colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#cbd5e1", "#7f7f7f"]

    # Determine max y for the right-hand bar chart across all steps so y-axis stays rock-solid
    all_bar_Series = [get_bar_counts_and_formula_annotation(i, step_labels[i])[0] for i in range(n_steps)]
    max_bar_y = int(max(max(vals) for vals in all_bar_Series) * 1.16) + 5

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.58, 0.42],
        subplot_titles=(
            "A. 2D Class Diagnostic Quadrant (Precision vs. Recall Trajectories)",
            f"B. Live Formula Component Counts ({'Class: ' + focus_target if focus_idx is not None else 'All 20 Classes'})"
        ),
        horizontal_spacing=0.12
    )

    # Preserve static subplot title annotations created by make_subplots
    base_annotations = list(fig.layout.annotations)

    # Trace 0: Static full trajectory rails (Col 1)
    fig.add_trace(
        go.Scatter(
            x=rail_x, y=rail_y,
            mode="lines",
            name="Full Trajectory Path",
            line=dict(color="rgba(148, 163, 184, 0.35)", width=1.5, dash="dot"),
            hoverinfo="skip",
            showlegend=True
        ),
        row=1, col=1
    )

    # Trace 1: Progressive active trail line (Col 1, mode='lines' only -> NO jumping markers!)
    tx0, ty0 = make_fixed_length_trail(0)
    fig.add_trace(
        go.Scatter(
            x=tx0, y=ty0,
            mode="lines",
            name="Traveled Trajectory",
            line=dict(color="rgba(37, 99, 235, 0.65)", width=2.4),
            hoverinfo="skip",
            showlegend=True
        ),
        row=1, col=1
    )

    # Trace 2: The 20 Class Bubbles (Col 1)
    def make_customdata(step_i: int):
        return np.stack([
            classes,
            gt_counts,
            m["tp_mat"][step_i],
            m["pure_fp_mat"][step_i],
            m["loc_err_mat"][step_i],
            m["mis_mat"][step_i],
            m["conf_drop_fn_mat"][step_i],
            m["blind_fn_mat"][step_i],
            m["total_fp_mat"][step_i],
            m["total_fn_mat"][step_i],
            m["f1_mat"][step_i]
        ], axis=-1)

    fig.add_trace(
        go.Scatter(
            x=m["rec_mat"][0],
            y=m["prec_mat"][0],
            ids=classes,
            mode="markers+text",
            name="VOC Classes (Size = GT)",
            text=make_bubble_labels(0),
            textposition="top right",
            textfont=dict(size=9.5, color="#0f172a"),
            marker=dict(
                size=bubble_sizes,
                color=c_mat[0],
                colorscale=cscale,
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                colorbar=dict(title=cbar_title, len=0.72, x=0.51, y=0.52, thickness=14),
                line=dict(width=line_widths, color=line_colors),
                opacity=0.92
            ),
            customdata=make_customdata(0),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (Total GT: %{customdata[1]})<br>"
                "<b>Precision: %{y:.1f}%</b> = TP (%{customdata[2]}) / [TP (%{customdata[2]}) + FP_total (%{customdata[8]})]<br>"
                "  ├─ Pure Background FP: %{customdata[3]}<br>"
                "  ├─ Localization Error (IoU < T): %{customdata[4]}<br>"
                "  └─ Misclassification: %{customdata[5]}<br>"
                "<b>Recall: %{x:.1f}%</b> = TP (%{customdata[2]}) / [TP (%{customdata[2]}) + FN_total (%{customdata[9]})]<br>"
                "  ├─ Filtered by Conf (< T): %{customdata[6]}<br>"
                "  ├─ Localization Error (IoU < T): %{customdata[4]}<br>"
                "  ├─ Misclassification: %{customdata[5]}<br>"
                "  └─ Blind Spot (IoU = 0): %{customdata[7]}<br>"
                "<b>F1-Score: %{customdata[10]:.1f}%</b><extra></extra>"
            )
        ),
        row=1, col=1
    )

    # Trace 3: Animated Formula Component Bar Chart (Col 2)
    bars0, formula0 = get_bar_counts_and_formula_annotation(0, step_labels[0])
    fig.add_trace(
        go.Bar(
            x=bar_categories,
            y=bars0,
            name="Formula Components",
            marker=dict(color=bar_colors, line=dict(color="#0f172a", width=1.0)),
            text=bars0,
            textposition="outside",
            textfont=dict(size=11, color="#0f172a"),
            hovertemplate="<b>%{x}</b><br>Count: <b>%{y}</b><extra></extra>",
            showlegend=False
        ),
        row=1, col=2
    )

    # Reference target lines in Col 1
    fig.add_shape(
        type="line", x0=80, x1=80, y0=0, y1=105,
        line=dict(color="rgba(100,116,139,0.5)", width=1.5, dash="dash"),
        xref="x1", yref="y1"
    )
    fig.add_shape(
        type="line", x0=0, x1=105, y0=70, y1=70,
        line=dict(color="rgba(100,116,139,0.5)", width=1.5, dash="dash"),
        xref="x1", yref="y1"
    )

    def make_frame_annotations(formula_html_str: str):
        live_box = dict(
            text=formula_html_str,
            xref="paper", yref="paper",
            x=0.5, y=-0.16,
            xanchor="center", yanchor="top",
            showarrow=False,
            align="left",
            font=dict(size=12, color="#0f172a"),
            bgcolor="rgba(241, 245, 249, 0.95)",
            bordercolor="#94a3b8",
            borderwidth=1.2,
            borderpad=8
        )
        return base_annotations + [live_box]

    frames = []
    for s_idx, lbl in enumerate(step_labels):
        tx_s, ty_s = make_fixed_length_trail(s_idx)
        bars_s, form_s = get_bar_counts_and_formula_annotation(s_idx, lbl)
        frames.append(go.Frame(
            name=lbl,
            data=[
                go.Scatter(x=tx_s, y=ty_s),
                go.Scatter(
                    x=m["rec_mat"][s_idx],
                    y=m["prec_mat"][s_idx],
                    ids=classes,
                    text=make_bubble_labels(s_idx),
                    marker=dict(
                        size=bubble_sizes,
                        color=c_mat[s_idx],
                        colorscale=cscale,
                        cmin=cmin,
                        cmax=cmax,
                        showscale=True,
                        colorbar=dict(title=cbar_title, len=0.72, x=0.51, y=0.52, thickness=14),
                        line=dict(width=line_widths, color=line_colors),
                        opacity=0.92
                    ),
                    customdata=make_customdata(s_idx)
                ),
                go.Bar(
                    x=bar_categories,
                    y=bars_s,
                    text=bars_s,
                    marker=dict(color=bar_colors, line=dict(color="#0f172a", width=1.0))
                )
            ],
            traces=[1, 2, 3],
            layout=go.Layout(
                annotations=make_frame_annotations(form_s)
            )
        ))

    fig.frames = frames

    sliders = [{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {"prefix": "Active Step: ", "visible": True, "xanchor": "right", "font": {"size": 13}},
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 115},
        "len": 0.86,
        "x": 0.14,
        "y": 0,
        "steps": [
            {
                "args": [[lbl], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                "label": lbl,
                "method": "animate"
            }
            for lbl in step_labels
        ]
    }]

    fig.update_layout(
        height=720,
        annotations=make_frame_annotations(formula0),
        xaxis1=dict(range=[-3, 105], title="True Positive Recall Rate (%)"),
        yaxis1=dict(range=[5, 105], title="Precision (%)"),
        xaxis2=dict(title="Formula Building Blocks", tickangle=-20),
        yaxis2=dict(range=[0, max_bar_y], title="Instance Count"),
        legend=dict(orientation="h", yanchor="bottom", y=1.07, xanchor="left", x=0.0),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0,
            "y": 0.0,
            "xanchor": "left",
            "yanchor": "top",
            "pad": {"t": 115, "r": 10},
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 140, "redraw": True}, "fromcurrent": True, "transition": {"duration": 80, "easing": "linear"}}]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                }
            ]
        }],
        sliders=sliders,
        margin=dict(l=25, r=25, t=85, b=195)
    )
    return fig


def build_animated_phase_space_and_scale_bars(
    df_in: pd.DataFrame,
    threshold_pairs: list,
    step_labels: list,
    strict_grey_iou: bool = False
) -> go.Figure:
    """
    Animation 2:
    Synchronized two-panel animation matching the Diagnostic Dashboard (Panel A + Panel B):
    - Left Subplot (Panel A): Joint Phase Space (Confidence vs. IoU across Scales) with moving decision gate lines
      and zero-jumping points that dynamically grey out or change state.
    - Right Subplot (Panel B): Scale Sensitivity Stacked Bar Chart (Small, Medium, Large Scale) whose bars
      grow and shrink live in sync with the threshold sweep!
    """
    base_d = evaluate_at_thresholds_vectorized(df_in, threshold_pairs[0][0], threshold_pairs[0][1], strict_grey_iou=strict_grey_iou)
    plot_base = base_d[base_d["Candidate_Confidence"].notna() & (~base_d["Is_Duplicate_Candidate_Row"])].copy()

    x_fixed = plot_base["Candidate_Confidence"].to_numpy()
    y_fixed = plot_base["IoU"].to_numpy()
    rec_ids = plot_base["Record_ID"].tolist()
    symbols_fixed = [SCALE_SYMBOLS.get(s, "circle") for s in plot_base["Environmental_Condition"]]

    # Non-duplicate rows used for Panel B scale bars
    scale_base = df_in[~df_in["Is_Duplicate_Candidate_Row"]].copy()
    active_scales = [s for s in SCALE_ORDER if s in scale_base["Environmental_Condition"].unique()]
    outcome_order = ["TP", "Misclassification", "Localization Error", "Missed Target", "False Positive"]
    outcome_bar_colors = {
        "TP": "#2ca02c",
        "Misclassification": "#d62728",
        "Localization Error": "#1f77b4",
        "Missed Target": "#7f7f7f",
        "False Positive": "#ff7f0e"
    }

    def compute_step_data(conf_t: float, iou_t: float):
        # 1. Panel A Scatter colors & opacities
        d_scatter = evaluate_at_thresholds_vectorized(plot_base, conf_t, iou_t, strict_grey_iou=strict_grey_iou)
        states = d_scatter["Phase_State"].tolist()
        colors = [PHASE_STATE_COLORS.get(st_name, "#cbd5e1") for st_name in states]
        opacities = [0.22 if st_name == "Filtered out (Greyed Out)" else 0.88 for st_name in states]
        sizes = [6.5 if st_name == "Filtered out (Greyed Out)" else 9.0 for st_name in states]
        custom = np.stack([
            d_scatter["Sample_ID"],
            d_scatter["Ground_Truth"],
            d_scatter["Effective_Pred_Class"],
            d_scatter["Environmental_Condition"],
            d_scatter["Phase_State"]
        ], axis=-1)

        # 2. Panel B Scale Proportions
        d_scale = evaluate_at_thresholds_vectorized(scale_base, conf_t, iou_t, strict_grey_iou=strict_grey_iou)
        d_scale_active = d_scale[d_scale["Dynamic_Outcome"] != "Filtered FP"]

        scale_pcts = {out: [] for out in outcome_order}
        scale_texts = {out: [] for out in outcome_order}

        for sc in active_scales:
            sc_sub = d_scale_active[d_scale_active["Environmental_Condition"] == sc]
            tot = max(1, len(sc_sub))
            for out in outcome_order:
                pct = (sc_sub["Dynamic_Outcome"] == out).sum() / tot * 100.0
                scale_pcts[out].append(round(pct, 1))
                scale_texts[out].append(f"{pct:.1f}%" if pct >= 3.5 else "")

        return colors, opacities, sizes, custom, scale_pcts, scale_texts

    c0, op0, sz0, cust0, pcts0, txts0 = compute_step_data(threshold_pairs[0][0], threshold_pairs[0][1])

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        subplot_titles=(
            "A. Joint Phase Space: Confidence vs. IoU across Scales",
            "B. Scale Sensitivity: Dynamic Outcome Proportions by Scale"
        ),
        horizontal_spacing=0.10
    )

    # Trace 0: Phase Space Scatter points in Col 1
    fig.add_trace(
        go.Scatter(
            x=x_fixed,
            y=y_fixed,
            ids=rec_ids,
            mode="markers",
            name="Phase Space Points",
            marker=dict(
                color=c0,
                opacity=op0,
                size=sz0,
                symbol=symbols_fixed,
                line=dict(width=0.6, color="white")
            ),
            customdata=cust0,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "GT: <b>%{customdata[1]}</b> | Pred/Cand: <b>%{customdata[2]}</b><br>"
                "Confidence: %{x:.3f} | IoU: %{y:.3f}<br>"
                "Scale: %{customdata[3]}<br>"
                "Current State: <b>%{customdata[4]}</b><extra></extra>"
            ),
            showlegend=False
        ),
        row=1, col=1
    )

    # Traces 1..5: The 5 Stacked Outcome Bars in Col 2 (also serve as main legend!)
    for out in outcome_order:
        fig.add_trace(
            go.Bar(
                x=active_scales,
                y=pcts0[out],
                name=out,
                marker=dict(color=outcome_bar_colors[out]),
                text=txts0[out],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="white", size=11),
                hovertemplate=f"<b>%{{x}}</b><br>{out}: <b>%{{y:.1f}}%</b><extra></extra>"
            ),
            row=1, col=2
        )

    # Trace 6: Legend entry for Greyed-Out points in Panel A
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            name="Filtered out (Greyed Out)",
            marker=dict(size=9, color="#cbd5e1")
        ),
        row=1, col=1
    )

    def make_gate_shapes(conf_t: float, iou_t: float):
        return [
            # Active acceptance region in Panel A
            dict(
                type="rect", x0=conf_t, x1=1.02, y0=iou_t, y1=1.03,
                fillcolor="rgba(44, 160, 44, 0.08)", line=dict(width=0),
                layer="below", xref="x1", yref="y1"
            ),
            # Greyed-out region left of Conf threshold in Panel A
            dict(
                type="rect", x0=-0.03, x1=conf_t, y0=-0.04, y1=1.03,
                fillcolor="rgba(148, 163, 184, 0.10)", line=dict(width=0),
                layer="below", xref="x1", yref="y1"
            ),
            # Vertical blue Confidence Gate line
            dict(
                type="line", x0=conf_t, x1=conf_t, y0=-0.04, y1=1.03,
                line=dict(color="#2563eb", width=2.5, dash="dot"),
                xref="x1", yref="y1"
            ),
            # Horizontal red IoU Gate line
            dict(
                type="line", x0=-0.03, x1=1.02, y0=iou_t, y1=iou_t,
                line=dict(color="#dc2626", width=2.5, dash="dash"),
                xref="x1", yref="y1"
            )
        ]

    frames = []
    for (c_t, i_t), lbl in zip(threshold_pairs, step_labels):
        cols, ops, szs, cust, pcts_s, txts_s = compute_step_data(c_t, i_t)
        frame_traces = [
            go.Scatter(
                x=x_fixed,
                y=y_fixed,
                ids=rec_ids,
                marker=dict(color=cols, opacity=ops, size=szs, symbol=symbols_fixed, line=dict(width=0.6, color="white")),
                customdata=cust
            )
        ]
        for out in outcome_order:
            frame_traces.append(
                go.Bar(
                    x=active_scales,
                    y=pcts_s[out],
                    text=txts_s[out],
                    marker=dict(color=outcome_bar_colors[out])
                )
            )

        frames.append(go.Frame(
            name=lbl,
            data=frame_traces,
            traces=[0, 1, 2, 3, 4, 5],
            layout=go.Layout(
                shapes=make_gate_shapes(c_t, i_t)
            )
        ))

    fig.frames = frames

    sliders = [{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {"prefix": "Active Gate: ", "visible": True, "xanchor": "right", "font": {"size": 13}},
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 40},
        "len": 0.86,
        "x": 0.14,
        "y": 0,
        "steps": [
            {
                "args": [[lbl], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                "label": lbl,
                "method": "animate"
            }
            for lbl in step_labels
        ]
    }]

    fig.update_layout(
        height=620,
        barmode="stack",
        shapes=make_gate_shapes(threshold_pairs[0][0], threshold_pairs[0][1]),
        xaxis1=dict(range=[-0.03, 1.02], title="Prediction / Candidate Confidence Score"),
        yaxis1=dict(range=[-0.04, 1.04], title="Intersection over Union (IoU with GT)"),
        xaxis2=dict(title="Object Scale Category (% of Image Area)"),
        yaxis2=dict(range=[0, 105], title="Percentage of Active Records (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5, font=dict(size=11)),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0,
            "y": 0.0,
            "xanchor": "left",
            "yanchor": "top",
            "pad": {"t": 40, "r": 10},
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 140, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                }
            ]
        }],
        sliders=sliders,
        margin=dict(l=25, r=25, t=95, b=90)
    )
    return fig


def apply_perturbation(image: Image.Image, brightness: float, contrast: float, noise_std: float) -> Image.Image:
    img_p = ImageEnhance.Brightness(image).enhance(brightness)
    img_p = ImageEnhance.Contrast(img_p).enhance(contrast)
    if noise_std > 0:
        rng = np.random.default_rng(seed=42)
        arr = np.array(img_p).astype(np.float32)
        arr = np.clip(arr + rng.normal(0, noise_std, arr.shape), 0, 255).astype(np.uint8)
        img_p = Image.fromarray(arr)
    return img_p


model = load_model()
eval_df = load_evaluation_data()

# ==============================================================================
# Header & Sidebar Navigation (100% English)
# ==============================================================================
st.title("YOLO26n Behavioral & Visual Analytics Studio")
st.caption(
    "Explainable Object Detection on Pascal VOC: Semantic Confusion Matrix, Smooth Multi-Panel Threshold "
    "Animations with Live Formula Breakdowns, and Single-Image Perturbation Trajectories."
)

st.sidebar.header("Navigation & Parameters")
analysis_mode = st.sidebar.radio(
    "Select Analysis Module",
    [
        "1. Confusion Matrix & Taxonomy Inspector",
        "2. Animated Diagnostic Studio (Trajectories & Phase Space)",
        "3. Single-Image Perturbation & NMS Trajectory Lab"
    ]
)

conf_threshold = st.sidebar.slider("Base Confidence Threshold (T_conf)", 0.25, 0.90, 0.25, step=0.05)
iou_threshold = st.sidebar.slider("Base IoU Match Threshold (T_IoU)", 0.15, 0.90, 0.50, step=0.05)

# ==============================================================================
# MODULE 1: Confusion Matrix & Taxonomy Shift Inspector (Streamlined)
# ==============================================================================
if analysis_mode == "1. Confusion Matrix & Taxonomy Inspector":
    if eval_df.empty:
        st.error("No `voc_behavioral_results.csv` found! Please run `python offline_test.py` first.")
    else:
        st.subheader("1. Interactive Semantic Confusion Matrix & Taxonomy Shift Inspector")

        with st.expander("ℹ️ Click to explain: How to Read the Confusion Matrix & Taxonomy Shifts (Simple Guide)", expanded=False):
            st.markdown(
                """
                * **What does the Confusion Matrix show?**  
                  Each **row** is the true object class annotated in the image (**Ground Truth**), and each **column** is what YOLO predicted (**Predicted Class**).
                * **Diagonal Cells (Top-Left to Bottom-Right):**  
                  Correct detections (**True Positives**). For example, Row = `dog` and Column = `dog`.
                * **Semantic Grouping (Animals / Vehicles / Indoor):**  
                  By ordering classes into semantic blocks, you can immediately see if mistakes stay within the same family (e.g., mistaking a `cow` or `sheep` for a `dog`) rather than random errors.
                * **Last Column (`None` = Missed Targets / False Negatives):**  
                  Real objects that the model failed to detect at the chosen Confidence and IoU thresholds.
                * **Last Row (`None (Background)` = False Positives):**  
                  Hallucinated boxes where YOLO predicted an object on empty background or produced a redundant box.
                * **COCO Out-of-Vocabulary (Taxonomy Shift):**  
                  Because `yolo26n.pt` was trained on MS COCO (80 classes) and evaluated on Pascal VOC (20 classes), it occasionally predicts a more specific COCO label (like `bear`, `zebra`, or `bench`) on top of a VOC object.
                """
            )

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            split_filter = st.multiselect(
                "Dataset Split:",
                options=list(eval_df["Split"].unique()),
                default=list(eval_df["Split"].unique())
            )
        with f_col2:
            scale_filter = st.multiselect(
                "Object Scale Condition:",
                options=[s for s in SCALE_ORDER if s in eval_df["Environmental_Condition"].unique()],
                default=[s for s in SCALE_ORDER if s in eval_df["Environmental_Condition"].unique()]
            )
        with f_col3:
            order_mode = st.radio(
                "Class Ordering:",
                ["Semantic Grouping (Animals / Vehicles / Indoor)", "Alphabetical"]
            )

        cm_opt1, cm_opt2 = st.columns(2)
        with cm_opt1:
            include_bg = st.checkbox("Include Background False Positives & Missed Targets", value=True)
        with cm_opt2:
            normalize_mode = st.radio(
                "Matrix Normalization:",
                ["Normalized by Ground Truth (Row Recall %)", "Raw Instance Counts"],
                horizontal=True
            )

        filtered_raw = eval_df[
            (eval_df["Split"].isin(split_filter)) &
            (eval_df["Environmental_Condition"].isin(scale_filter))
        ].copy()

        filtered = evaluate_at_thresholds_vectorized(filtered_raw, conf_threshold, iou_threshold, strict_grey_iou=False)

        active_classes = VOC_CLASSES_SEMANTIC if "Semantic" in order_mode else VOC_CLASSES_ALPHABETICAL
        y_classes = list(active_classes) + (["None (Background)"] if include_bg else [])
        x_classes = list(active_classes) + (["None"] if include_bg else [])

        cm_records = filtered[~filtered["Is_Duplicate_Candidate_Row"]].copy()
        cm_records["CM_Pred"] = np.where(
            cm_records["Dynamic_Outcome"].isin(["TP", "Misclassification", "False Positive"]),
            cm_records["Effective_Pred_Class"],
            "None"
        )
        cm_records = cm_records[cm_records["Dynamic_Outcome"] != "Filtered FP"]

        matrix_df = cm_records[
            (cm_records["Ground_Truth"].isin(y_classes)) &
            (cm_records["CM_Pred"].isin(x_classes))
        ]

        cm = pd.crosstab(
            matrix_df["Ground_Truth"],
            matrix_df["CM_Pred"],
            dropna=False
        ).reindex(index=y_classes, columns=x_classes, fill_value=0)

        if "Normalized" in normalize_mode:
            row_sums = cm.sum(axis=1)
            cm_plot = (cm.div(row_sums.replace(0, 1), axis=0) * 100).round(1)
            color_label = "Row Percentage (%)"
        else:
            cm_plot = cm
            color_label = "Instance Count"

        fig_cm = px.imshow(
            cm_plot,
            labels=dict(x="Predicted Class (Active Model Output)", y="Ground Truth (Dataset Annotation)", color=color_label),
            x=x_classes,
            y=y_classes,
            color_continuous_scale="Blues",
            text_auto=True,
            aspect="auto"
        )
        fig_cm.update_layout(
            title=f"Diagnostic Confusion Matrix (Active at Conf ≥ {conf_threshold:.2f}, IoU ≥ {iou_threshold:.2f})",
            height=680,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        oov_df = filtered[
            (filtered["Dynamic_Outcome"] == "Misclassification") &
            (~filtered["Effective_Pred_Class"].isin(VOC_CLASSES_ALPHABETICAL))
        ]
        if not oov_df.empty:
            with st.expander(f"🔎 Inspect {len(oov_df)} COCO Out-of-Vocabulary (Taxonomy Shift) Predictions", expanded=False):
                st.markdown(
                    "These Ground Truth objects were localized accurately (`IoU ≥ T_IoU`), but the model assigned "
                    "a MS-COCO class label that does not exist in the 20 Pascal VOC categories:"
                )
                st.dataframe(
                    oov_df[["Sample_ID", "Ground_Truth", "Effective_Pred_Class", "Candidate_Confidence", "IoU", "Environmental_Condition"]],
                    use_container_width=True
                )


# ==============================================================================
# MODULE 2: Animated Diagnostic Studio (Class Trajectories + Live Formula + Phase Space & Scale Bars)
# ==============================================================================
elif analysis_mode == "2. Animated Diagnostic Studio (Trajectories & Phase Space)":
    st.subheader("2. Animated Diagnostic Studio: Precision-Recall Trajectories, Live Formulas & Phase Space")

    with st.expander("ℹ️ Click to explain: Precision, Recall, IoU & All Error Types (With Simple Examples)", expanded=False):
        st.markdown(
            """
            ### 1. Core Metrics Explained Simply
            * **Intersection over Union ($\text{IoU}$):**  
              Measures how accurately a predicted box overlaps the true object box ($0.0$ = no overlap at all, $1.0$ = pixel-perfect match).  
              *Example:* At $\text{IoU Threshold } T_{\text{IoU}} = 0.50$, a box needs at least $50\%$ overlap to count as a valid hit. If you raise $T_{\text{IoU}}$ to $0.85$, a box with $0.80$ overlap is no longer accurate enough!
            * **Precision ($\text{When the model predicts a dog, how often is it really an accurate dog?}$):**  
              $$\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}_{\\text{total}}} = \\frac{\\text{TP}}{\\text{TP} + \\text{Pure BG FP} + \\text{Loc. Error} + \\text{Misclass}}$$
              *Example:* YOLO predicts **49 boxes** for `dog`. At $T_{\text{IoU}} = 0.50$, **37** are accurate hits ($\text{TP}$) and **12** are false alarms ($\text{FP}$), so $\\text{Precision} = 37 / 49 = 75.5\\%$.  
              **Why does Precision drop when $T_{\text{IoU}}$ increases?** The model still predicts the exact same 49 boxes! But at $T_{\text{IoU}} = 0.90$, 11 previously valid dog boxes fail the stricter overlap test and turn from $\text{TP}$ into **Localization Errors ($\text{FP}$)**. Now only $26 / 49 = 53.1\\%$ of the predicted boxes are valid hits!
            * **Recall ($\text{Out of all real dogs in the dataset, what percentage did the model find?}$):**  
              $$\\text{Recall} = \\frac{\\text{TP}}{\\text{Total Ground Truth}} = \\frac{\\text{TP}}{\\text{TP} + \\text{Missed (Conf < T)} + \\text{Loc. Error} + \\text{Misclass} + \\text{Blind Spot}}$$
              *Example:* There are **46 real dogs** in the dataset. Finding 37 of them gives $\\text{Recall} = 37 / 46 = 80.4\\%$. Raising the Confidence Threshold $T_{\text{conf}}$ filters out low-confidence boxes, which eliminates False Positives ($\text{Precision} \\uparrow$) but also drops real detections ($\text{Recall} \\downarrow$).

            ---
            ### 2. Fine-Grained Outcome Categories in the Bar Charts & Phase Space
            * 🟢 **Active TP (True Positive):** Correct class label, $\\text{Confidence} \\ge T_{\\text{conf}}$, and $\\text{IoU} \\ge T_{\\text{IoU}}$. Increases both Precision and Recall.
            * 🟠 **Pure Background FP (False Positive):** The model hallucinated an object on background or produced a redundant box. Increases the denominator of Precision ($\text{Precision} \\downarrow$).
            * 🔵 **Localization Error ($\text{Conf} \\ge T_{\\text{conf}}$, but $0 < \\text{IoU} < T_{\\text{IoU}}$):** The model detected an object, but the box is too shifted or imprecise to pass $T_{\\text{IoU}}$. Hurts **both** Precision (counts as an inaccurate box) and Recall (the true object remains unmatched).
            * 🔴 **Misclassification ($\text{Conf} \\ge T_{\\text{conf}}$, $\\text{IoU} \\ge T_{\\text{IoU}}$, wrong class):** Good box location, but wrong class label (e.g., predicting `cat` on a `dog`).
            * ⚪ **Filtered out / Missed ($\text{Conf} < T_{\\text{conf}}$):** Box confidence fell below the Confidence Threshold, so the box is suppressed and greyed out.
            * ⚫ **Blind Spot ($\text{IoU} = 0.0$):** A true object where YOLO did not place any overlapping candidate box at all.
            """
        )

    if eval_df.empty:
        st.error("No `voc_behavioral_results.csv` found! Please run `python offline_test.py` first.")
    else:
        # Top Simulation Controls
        c_top1, c_top2, c_top3 = st.columns([1.5, 1.1, 1.2])
        with c_top1:
            sweep_mode = st.radio(
                "1. Select Parameter Sweep to Animate:",
                [
                    f"Confidence Threshold Sweep (0.25 → 0.90 at fixed IoU = {iou_threshold:.2f})",
                    f"IoU Match Requirement Sweep (0.15 → 0.90 at fixed Conf = {conf_threshold:.2f})",
                    "Joint Strictness Sweep (Simultaneous Conf 0.25→0.85 & IoU 0.25→0.85)"
                ]
            )
        with c_top2:
            step_res_label = st.selectbox(
                "2. Step Resolution (Smoothness):",
                [
                    "0.01 (Ultra-Smooth: 0.15, 0.16, 0.17...)",
                    "0.02 (Fast & Smooth: 0.15, 0.17, 0.19...)",
                    "0.05 (Coarse Steps: 0.15, 0.20, 0.25...)"
                ],
                index=0
            )
            step_size = 0.01 if "0.01" in step_res_label else (0.02 if "0.02" in step_res_label else 0.05)
        with c_top3:
            anim_scale_filter = st.multiselect(
                "3. Filter Object Scales:",
                options=SCALE_ORDER,
                default=SCALE_ORDER
            )

        sub_anim_df = eval_df[eval_df["Environmental_Condition"].isin(anim_scale_filter)].copy()

        if "Confidence Threshold Sweep" in sweep_mode:
            c_steps = np.round(np.arange(0.25, 0.90 + 1e-6, step_size), 2)
            threshold_pairs = [(float(c), float(iou_threshold)) for c in c_steps]
            step_labels = [f"Conf≥{c:.2f}" for c in c_steps]
        elif "IoU Match Requirement Sweep" in sweep_mode:
            i_steps = np.round(np.arange(0.15, 0.90 + 1e-6, step_size), 2)
            threshold_pairs = [(float(conf_threshold), float(i)) for i in i_steps]
            step_labels = [f"IoU≥{i:.2f}" for i in i_steps]
        else:
            n_pts = int(round((0.85 - 0.25) / step_size)) + 1
            c_steps = np.round(np.linspace(0.25, 0.85, n_pts), 2)
            i_steps = np.round(np.linspace(0.25, 0.85, n_pts), 2)
            threshold_pairs = [(float(c), float(i)) for c, i in zip(c_steps, i_steps)]
            step_labels = [f"C≥{c:.2f},IoU≥{i:.2f}" for c, i in zip(c_steps, i_steps)]

        # ======================================================================
        # ANIMATION 1: 2D Class Diagnostic Quadrant + Formula Bars + Live Equations
        # ======================================================================
        st.markdown("---")
        st.markdown("### Animation 1: 2D Class Diagnostic Quadrant (Precision vs. Recall) & Live Formula Breakdown")

        q_col1, q_col2, q_col3 = st.columns([1.4, 1.2, 1.1])
        with q_col1:
            bubble_color_mode = st.selectbox(
                "Bubble Color Encoding (Per-Class Metric):",
                [
                    "Error Share in Predictions: FP_total / (TP + FP_total) [%]",
                    "Active False Alarms Count (Pure FP + Loc.Err + Misclass)",
                    "Missed Target Share: FN_total / GT_Count [%]",
                    "Class F1-Score (%)"
                ]
            )
        with q_col2:
            focus_target = st.selectbox(
                "Inspect Formula & Bar Chart Scope (Right Panel):",
                ["All 20 Classes (Dataset Total)"] + VOC_CLASSES_ALPHABETICAL,
                index=0,
                help="Select 'All 20 Classes' or pick a specific class (e.g., 'dog', 'chair', 'person') to watch its exact TP, FP, and FN bars and formula numbers update live!"
            )
        with q_col3:
            show_live_counts_on_bubbles = st.checkbox(
                "Show live [TP|FP|FN] counts on every bubble label",
                value=False,
                help="Displays each class's live TP, FP, and FN numbers directly next to its bubble in the 2D scatter plot."
            )

        fig_quad_anim = build_animated_quadrant_and_formula_bars(
            sub_anim_df,
            threshold_pairs=threshold_pairs,
            step_labels=step_labels,
            color_mode=bubble_color_mode,
            focus_target=focus_target,
            show_counts_on_labels=show_live_counts_on_bubbles
        )
        st.plotly_chart(fig_quad_anim, use_container_width=True)

        # Mathematical Formula Reference Box right underneath Animation 1
        st.info(
            "**Mathematical Reference (Updated Live Inside the Animation Box Above):**\n"
            "$$\\text{Precision}(T_{\\text{conf}}, T_{\\text{IoU}}) = \\frac{\\text{Active TP}}{\\text{Active TP} + \\underbrace{\\text{Pure BG FP} + \\text{Loc. Error } (0 < \\text{IoU} < T_{\\text{IoU}}) + \\text{Misclass}}_{\\text{Total Active False Alarms } (\\text{FP}_{\\text{total}})}} "
            "\\qquad\\qquad "
            "\\text{Recall}(T_{\\text{conf}}, T_{\\text{IoU}}) = \\frac{\\text{Active TP}}{\\underbrace{\\text{Active TP} + \\text{Missed } (\\text{Conf} < T_{\\text{conf}}) + \\text{Loc. Error} + \\text{Misclass} + \\text{Blind Spot}}_{\\text{Total Ground Truth Objects } (\\text{GT}_{\\text{total}})}}$$"
        )

        # ======================================================================
        # ANIMATION 2: Synchronized Phase Space (Panel A) & Scale Bars (Panel B)
        # ======================================================================
        st.markdown("---")
        st.markdown("### Animation 2: Synchronized Joint Phase Space (Panel A) & Scale Sensitivity Proportions (Panel B)")
        st.caption(
            "Press **▶ Play** or drag the slider (`0.15, 0.16, 0.17...`): On the left, the threshold decision gate moves across "
            "the fixed points in the `(Confidence, IoU)` phase space and greys out filtered detections. On the right, the stacked "
            "percentage bars for Small, Medium, and Large objects dynamically grow and shrink in real time."
        )

        strict_gate_toggle = st.checkbox(
            "Strict Gate Mode in Phase Space (Grey out points below T_IoU instead of highlighting Localization Errors in blue)",
            value=False
        )

        fig_phase_scale_anim = build_animated_phase_space_and_scale_bars(
            sub_anim_df,
            threshold_pairs=threshold_pairs,
            step_labels=step_labels,
            strict_grey_iou=strict_gate_toggle
        )
        st.plotly_chart(fig_phase_scale_anim, use_container_width=True)


# ==============================================================================
# MODULE 3: Single-Image Perturbation & NMS Trajectory Lab
# ==============================================================================
else:
    st.subheader("3. Single-Image Perturbation & NMS Trajectory Lab")

    with st.expander("ℹ️ Click to explain: Input Perturbations & Non-Maximum Suppression (NMS)", expanded=False):
        st.markdown(
            """
            * **Input Perturbations (Brightness, Contrast, Gaussian Noise):**  
              Tests how robust YOLO's feature extractor is against real-world sensor degradation (e.g., low-light scenes, overexposure, or grainy camera noise).
            * **Confidence Threshold ($T_{\\text{conf}}$):**  
              Discards any candidate bounding box whose predicted class probability is below $T_{\\text{conf}}$.
            * **NMS IoU Threshold ($T_{\\text{NMS}}$):**  
              Object detectors often predict multiple overlapping boxes around the same object. **Non-Maximum Suppression (NMS)** keeps the highest-confidence box and deletes any neighboring box of the same class whose overlap with it exceeds $T_{\\text{NMS}}$.
            """
        )

    uploaded_file = st.sidebar.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    candidate_img_dirs = [
        BASE_DIR / "VOC_dataset" / "VOCdevkit" / "VOC2007" / "JPEGImages",
        BASE_DIR / "archive" / "VOC_dataset" / "VOCdevkit" / "VOC2007" / "JPEGImages",
    ]
    voc_sample_dir = next((d for d in candidate_img_dirs if d.exists() and any(d.glob("*.jpg"))), None)

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")
    elif voc_sample_dir is not None:
        sample_list = sorted([p.name for p in list(voc_sample_dir.glob("*.jpg"))[:50]])
        selected_sample = st.sidebar.selectbox("Or choose a VOC2007 sample:", sample_list)
        raw_image = Image.open(voc_sample_dir / selected_sample).convert("RGB")
    else:
        raw_image = Image.new("RGB", (640, 480), color=(120, 130, 140))
        st.info("Upload an image in the sidebar or place Pascal VOC images in `VOC_dataset/VOCdevkit/VOC2007/JPEGImages`.")

    st.sidebar.subheader("Manual Perturbation Controls")
    b_val = st.sidebar.slider("Brightness Factor", 0.2, 2.5, 1.0, 0.1)
    c_val = st.sidebar.slider("Contrast Factor", 0.2, 2.5, 1.0, 0.1)
    noise_val = st.sidebar.slider("Gaussian Noise (σ)", 0, 60, 0, 5)

    img_perturbed = apply_perturbation(raw_image, b_val, c_val, noise_val)
    results = model.predict(img_perturbed, conf=conf_threshold, iou=iou_threshold, verbose=False)
    res = results[0]

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown(f"**Perturbed Input** (`Brightness={b_val}`, `Contrast={c_val}`, `Noise σ={noise_val}`)")
        st.image(img_perturbed, use_container_width=True)
    with img_col2:
        st.markdown(f"**YOLO26n Detections** (`conf ≥ {conf_threshold:.2f}`, `IoU NMS = {iou_threshold:.2f}`)")
        st.image(res.plot()[:, :, ::-1], use_container_width=True)

    detections = []
    img_w, img_h = img_perturbed.size
    total_area = float(max(1, img_w * img_h))

    if res.boxes is not None and len(res.boxes) > 0:
        for idx, b in enumerate(res.boxes):
            cid = int(b.cls[0].item())
            cls_name = normalize_label(res.names.get(cid, str(cid)))
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            box_area_pct = ((x2 - x1) * (y2 - y1)) / total_area * 100.0
            detections.append({
                "Box_ID": f"Obj_{idx+1} ({cls_name})",
                "Class": cls_name,
                "Confidence": round(float(b.conf[0].item()), 3),
                "Relative Area (%)": round(box_area_pct, 2),
                "Center X": round((x1 + x2) / 2, 1),
                "Center Y": round((y1 + y2) / 2, 1),
                "BBox [x1, y1, x2, y2]": [round(v, 1) for v in [x1, y1, x2, y2]]
            })

    if detections:
        det_df = pd.DataFrame(detections)
        d_col1, d_col2 = st.columns([1.2, 1])
        with d_col1:
            st.dataframe(det_df, use_container_width=True)
        with d_col2:
            fig_det = px.scatter(
                det_df,
                x="Relative Area (%)",
                y="Confidence",
                color="Class",
                size="Relative Area (%)",
                text="Box_ID",
                range_y=[0, 1.05],
                title="Single-Image Detection Scale vs. Confidence"
            )
            fig_det.update_traces(textposition="top center")
            st.plotly_chart(fig_det, use_container_width=True)
    else:
        st.warning("No objects detected above the selected Confidence Threshold.")

    st.markdown("---")
    st.markdown("### Automated Perturbation Robustness Trajectory (Live Sweep)")
    sw_col1, sw_col2 = st.columns([1, 2])
    with sw_col1:
        pert_axis = st.selectbox(
            "Select Perturbation Axis to Sweep:",
            ["Gaussian Noise σ (0 → 60)", "Brightness Factor (0.2 → 2.2)", "Contrast Factor (0.2 → 2.2)"]
        )
        run_sweep = st.button("▶ Run Live Perturbation Sweep", type="primary")

    if run_sweep:
        sweep_records = []
        if "Noise" in pert_axis:
            steps = np.arange(0, 65, 5)
            param_name = "Noise σ"
        elif "Brightness" in pert_axis:
            steps = np.round(np.arange(0.2, 2.3, 0.2), 2)
            param_name = "Brightness"
        else:
            steps = np.round(np.arange(0.2, 2.3, 0.2), 2)
            param_name = "Contrast"

        progress = st.progress(0)
        for i, val in enumerate(steps):
            if "Noise" in pert_axis:
                sim_img = apply_perturbation(raw_image, 1.0, 1.0, float(val))
            elif "Brightness" in pert_axis:
                sim_img = apply_perturbation(raw_image, float(val), 1.0, 0.0)
            else:
                sim_img = apply_perturbation(raw_image, 1.0, float(val), 0.0)

            r_sim = model.predict(sim_img, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
            if r_sim.boxes is not None and len(r_sim.boxes) > 0:
                for b in r_sim.boxes:
                    cid = int(b.cls[0].item())
                    cname = normalize_label(r_sim.names.get(cid, str(cid)))
                    sweep_records.append({
                        param_name: float(val),
                        "Class": cname,
                        "Confidence": float(b.conf[0].item())
                    })
            else:
                sweep_records.append({
                    param_name: float(val),
                    "Class": "None (No Detection)",
                    "Confidence": 0.0
                })
            progress.progress((i + 1) / len(steps))

        sw_df = pd.DataFrame(sweep_records)
        agg_sw = (
            sw_df[sw_df["Class"] != "None (No Detection)"]
            .groupby([param_name, "Class"], as_index=False)
            .agg(Max_Confidence=("Confidence", "max"), Mean_Confidence=("Confidence", "mean"), Count=("Confidence", "count"))
        )

        if not agg_sw.empty:
            fig_traj = px.line(
                agg_sw,
                x=param_name,
                y="Max_Confidence",
                color="Class",
                markers=True,
                hover_data=["Count", "Mean_Confidence"],
                range_y=[0, 1.05],
                title=f"Detection Confidence Trajectory across {param_name}"
            )
            fig_traj.add_hline(y=conf_threshold, line_dash="dot", line_color="red", annotation_text="Confidence Cutoff")
            st.plotly_chart(fig_traj, use_container_width=True)
        else:
            st.warning("No objects were detected across the entire perturbation sweep.")