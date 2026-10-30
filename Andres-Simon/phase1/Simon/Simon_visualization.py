from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Unified 5 Dynamic Outcome Names & Colors across Animation 1 and Animation 2
OUTCOME_NAMES_5 = [
    "TP (Conf, IoU>T)",
    "FP (Haluzination)",
    "Missed (IoU<T)",
    "Misclassification",
    "Missed (Conf<T)"
]

OUTCOME_COLORS_5 = {
    "TP (Conf, IoU>T)": "#2ca02c",
    "FP (Haluzination)": "#ff7f0e",
    "Missed (IoU<T)": "#1f77b4",
    "Misclassification": "#d62728",
    "Missed (Conf<T)": "#94a3b8"
}

# 5 harmonious, eye-friendly continuous gradients (v=0.0 light pastel -> v=1.0 rich saturated tone)
# Designed so dark text (#0f172a) stays 100% readable across the entire gradient without ever flipping!
ROW_GRADIENT_PALETTES = [
    # Row 0: TP (Conf, IoU>T) -> Soft Mint to Rich Forest Sage
    ["#ecfdf5", "#a7f3d0", "#4ade80", "#16a34a"],
    # Row 1: FP (Haluzination) -> Soft Warm Cream to Rich Amber/Terracotta
    ["#fffbeb", "#fde68a", "#fb923c", "#ea580c"],
    # Row 2: Missed (IoU<T) -> Soft Ice Blue to Rich Steel/Cobalt Blue
    ["#f0f9ff", "#bae6fd", "#60a5fa", "#2563eb"],
    # Row 3: Misclassification -> Soft Blush Rose to Rich Crimson/Rose
    ["#fff1f2", "#fecdd3", "#fb7185", "#e11d48"],
    # Row 4: Missed (Conf<T) -> Soft Alabaster to Medium-Deep Cool Slate
    ["#f8fafc", "#e2e8f0", "#94a3b8", "#64748b"],
]

ROW_CMAPS = [
    mcolors.LinearSegmentedColormap.from_list(f"row_cmap_{i}", pal, N=512)
    for i, pal in enumerate(ROW_GRADIENT_PALETTES)
]

SCALE_ORDER = [
    "Small Scale (<4%)",
    "Medium Scale (4-20%)",
    "Large Scale (>20%)"
]

SCALE_SHORT_LABELS = [
    "Small (<4%)",
    "Medium (4-20%)",
    "Large (>20%)"
]

SCALE_SYMBOLS = {
    "Small Scale (<4%)": "square",
    "Medium Scale (4-20%)": "circle",
    "Large Scale (>20%)": "x"
}


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
    iou_t: float
) -> pd.DataFrame:
    """
    Fast vectorized evaluation at (conf_t, iou_t) using unified naming:
    - 'TP (Conf, IoU>T)'
    - 'FP (Haluzination)'
    - 'Missed (IoU<T)'
    - 'Misclassification'
    - 'Missed (Conf<T)'
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

    dyn_out = np.full(len(d), "Blind Spot", dtype=object)
    gt_active = is_gt & has_box & pass_conf
    dyn_out[gt_active & pass_iou & valid_cls & same_cls] = "TP (Conf, IoU>T)"
    dyn_out[gt_active & pass_iou & valid_cls & (~same_cls)] = "Misclassification"
    dyn_out[gt_active & (~(pass_iou & valid_cls))] = "Missed (IoU<T)"
    dyn_out[is_gt & has_box & (~pass_conf)] = "Missed (Conf<T)"

    fp_rows = ~is_gt
    dyn_out[fp_rows & (~pass_conf)] = "Filtered FP"
    dyn_out[fp_rows & pass_conf] = "FP (Haluzination)"

    p_state = np.full(len(d), "Missed (Conf<T)", dtype=object)
    p_state[is_gt & (~has_box)] = "Blind Spot"
    p_state[gt_active & pass_iou & valid_cls & same_cls] = "TP (Conf, IoU>T)"
    p_state[gt_active & pass_iou & valid_cls & (~same_cls)] = "Misclassification"
    p_state[gt_active & (~(pass_iou & valid_cls))] = "Missed (IoU<T)"
    p_state[fp_rows & pass_conf] = "FP (Haluzination)"

    d["Dynamic_Outcome"] = dyn_out
    d["Phase_State"] = p_state
    d["Effective_Pred_Class"] = eff_cls
    return d


def compute_all_steps_class_and_scale_metrics(df_in: pd.DataFrame, threshold_pairs: list) -> dict:
    """
    Vectorized computation across all threshold steps (0.01 resolution):
    - Per-class Precision, Recall, F1, and the 5 outcome matrices (shape: n_steps x 20)
    - 3D Scale x Outcome matrices (shape: n_steps x 3 scales x 5 outcomes)
    - Smoothly interpolated continuous color hex matrices for the 5x20 table so colors never jump or blink
    """
    orig = df_in["Outcome"].to_numpy()
    gt_cls = df_in["Ground_Truth"].to_numpy()
    cand_cls = df_in["Candidate_Class"].to_numpy()
    pred_cls = df_in["Predicted_Class"].to_numpy()
    eff_cls = np.where(cand_cls != "None", cand_cls, pred_cls)
    conf = df_in["Candidate_Confidence"].fillna(-1.0).to_numpy()
    iou = df_in["IoU"].fillna(0.0).to_numpy()
    is_dup = df_in["Is_Duplicate_Candidate_Row"].to_numpy()
    scale_arr = df_in["Environmental_Condition"].to_numpy()

    is_gt = orig != "False Positive"
    is_pure_fp_row = (orig == "False Positive") & (~is_dup)
    has_box = conf >= 0.0

    cls_list = VOC_CLASSES_ALPHABETICAL
    n_cls = len(cls_list)
    n_steps = len(threshold_pairs)
    n_scales = len(SCALE_ORDER)

    gt_cls_masks = np.array([is_gt & (gt_cls == c) for c in cls_list])
    eff_cls_gt_masks = np.array([is_gt & (eff_cls == c) for c in cls_list])
    fp_cls_masks = np.array([is_pure_fp_row & (pred_cls == c) for c in cls_list])
    scale_masks = np.array([scale_arr == sc for sc in SCALE_ORDER])

    gt_counts = gt_cls_masks.sum(axis=1)
    blind_counts = np.array([(gt_cls_masks[i] & (~has_box)).sum() for i in range(n_cls)])

    rec_mat = np.zeros((n_steps, n_cls))
    prec_mat = np.zeros((n_steps, n_cls))
    f1_mat = np.zeros((n_steps, n_cls))

    tp_mat = np.zeros((n_steps, n_cls), dtype=int)
    pure_fp_mat = np.zeros((n_steps, n_cls), dtype=int)
    loc_err_mat = np.zeros((n_steps, n_cls), dtype=int)
    mis_mat = np.zeros((n_steps, n_cls), dtype=int)
    conf_drop_fn_mat = np.zeros((n_steps, n_cls), dtype=int)
    blind_fn_mat = np.tile(blind_counts, (n_steps, 1))

    # Continuous soft-gated matrices for silky-smooth color gradients in the 5x20 table
    soft_tp_mat = np.zeros((n_steps, n_cls), dtype=float)
    soft_fp_mat = np.zeros((n_steps, n_cls), dtype=float)
    soft_loc_mat = np.zeros((n_steps, n_cls), dtype=float)
    soft_mis_mat = np.zeros((n_steps, n_cls), dtype=float)
    soft_conf_drop_mat = np.zeros((n_steps, n_cls), dtype=float)

    global_3d_bars = np.zeros((n_steps, n_scales, 5), dtype=int)
    class_3d_bars = np.zeros((n_steps, n_cls, n_scales, 5), dtype=int)

    valid_cls = eff_cls != "None"
    same_cls = eff_cls == gt_cls
    tau = 0.055  # Soft sigmoid transition bandwidth for continuous color gradient

    for s_idx, (c_t, i_t) in enumerate(threshold_pairs):
        pass_conf = conf >= c_t
        pass_iou = iou >= i_t

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

        tp_mat[s_idx] = tp_c
        pure_fp_mat[s_idx] = fp_c
        loc_err_mat[s_idx] = loc_c
        mis_mat[s_idx] = mis_c
        conf_drop_fn_mat[s_idx] = conf_drop_c

        # Soft sigmoid weights in [0, 1] so every 0.01 step produces a smooth continuous color shift
        w_conf = np.where(has_box, 1.0 / (1.0 + np.exp(-np.clip((conf - c_t) / tau, -20.0, 20.0))), 0.0)
        w_iou = np.where(has_box, 1.0 / (1.0 + np.exp(-np.clip((iou - i_t) / tau, -20.0, 20.0))), 0.0)

        soft_tp_w = np.where(is_gt & has_box & valid_cls & same_cls, w_conf * w_iou, 0.0)
        soft_mis_w = np.where(is_gt & has_box & valid_cls & (~same_cls), w_conf * w_iou, 0.0)
        soft_loc_w = np.where(is_gt & has_box, w_conf * (1.0 - w_iou), 0.0)
        soft_conf_drop_w = np.where(is_gt & has_box, 1.0 - w_conf, 0.0)
        soft_fp_w = np.where(is_pure_fp_row, w_conf, 0.0)

        soft_tp_mat[s_idx] = gt_cls_masks @ soft_tp_w
        soft_mis_mat[s_idx] = gt_cls_masks @ soft_mis_w
        soft_loc_mat[s_idx] = eff_cls_gt_masks @ soft_loc_w
        soft_conf_drop_mat[s_idx] = gt_cls_masks @ soft_conf_drop_w
        soft_fp_mat[s_idx] = fp_cls_masks @ soft_fp_w

        for sc_i in range(n_scales):
            sm = scale_masks[sc_i]
            global_3d_bars[s_idx, sc_i, 0] = int((tp_mask & sm).sum())
            global_3d_bars[s_idx, sc_i, 1] = int((fp_active_mask & sm).sum())
            global_3d_bars[s_idx, sc_i, 2] = int((loc_mask & sm).sum())
            global_3d_bars[s_idx, sc_i, 3] = int((mis_mask & sm).sum())
            global_3d_bars[s_idx, sc_i, 4] = int((conf_drop_mask & sm).sum())

            class_3d_bars[s_idx, :, sc_i, 0] = (gt_cls_masks & (tp_mask & sm)).sum(axis=1)
            class_3d_bars[s_idx, :, sc_i, 1] = (fp_cls_masks & (fp_active_mask & sm)).sum(axis=1)
            class_3d_bars[s_idx, :, sc_i, 2] = (eff_cls_gt_masks & (loc_mask & sm)).sum(axis=1)
            class_3d_bars[s_idx, :, sc_i, 3] = (gt_cls_masks & (mis_mask & sm)).sum(axis=1)
            class_3d_bars[s_idx, :, sc_i, 4] = (gt_cls_masks & (conf_drop_mask & sm)).sum(axis=1)

    # Normalize soft intensities per class & row with smooth damping so 1-2 count changes never jump 100%
    gt_cap = np.maximum(1.0, gt_counts.astype(float))
    fp_max_c = np.maximum(1.0, soft_fp_mat.max(axis=0))
    loc_max_c = np.maximum(1.0, soft_loc_mat.max(axis=0))
    mis_max_c = np.maximum(1.0, soft_mis_mat.max(axis=0))

    v_raw = np.zeros((n_steps, 5, n_cls), dtype=float)
    v_raw[:, 0, :] = np.clip(soft_tp_mat / gt_cap, 0.0, 1.0)
    v_raw[:, 1, :] = np.clip((soft_fp_mat / fp_max_c) * (fp_max_c / (fp_max_c + 2.5)), 0.0, 1.0)
    v_raw[:, 2, :] = np.clip((soft_loc_mat / loc_max_c) * (loc_max_c / (loc_max_c + 2.0)), 0.0, 1.0)
    v_raw[:, 3, :] = np.clip((soft_mis_mat / mis_max_c) * (mis_max_c / (mis_max_c + 2.0)), 0.0, 1.0)
    v_raw[:, 4, :] = np.clip(soft_conf_drop_mat / gt_cap, 0.0, 1.0)

    # Additional temporal Gaussian smoothing along the sweep axis so every frame is a fluid gradient
    v_smooth = np.clip(gaussian_filter1d(v_raw, sigma=3.5, axis=0, mode="nearest"), 0.0, 1.0)

    return {
        "classes": cls_list,
        "gt_counts": gt_counts,
        "rec_mat": rec_mat,
        "prec_mat": prec_mat,
        "f1_mat": f1_mat,
        "tp_mat": tp_mat,
        "pure_fp_mat": pure_fp_mat,
        "loc_err_mat": loc_err_mat,
        "mis_mat": mis_mat,
        "conf_drop_fn_mat": conf_drop_fn_mat,
        "blind_fn_mat": blind_fn_mat,
        "total_fp_mat": pure_fp_mat + loc_err_mat + mis_mat,
        "total_fn_mat": conf_drop_fn_mat + loc_err_mat + mis_mat + blind_fn_mat,
        "global_3d_bars": global_3d_bars,
        "class_3d_bars": class_3d_bars,
        "v_smooth": v_smooth,
    }


def build_cuboid_mesh_arrays(x_center: float, y_centers: list, z_heights: list, dx: float = 0.26, dy: float = 0.26):
    """
    Creates (x, y, z, i, j, k) arrays for 3 cuboids (one per scale along Y) at a fixed X category index.
    """
    xs, ys, zs = [], [], []
    ii, jj, kk = [], [], []
    base_i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    base_j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    base_k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]

    for idx, (yc, h) in enumerate(zip(y_centers, z_heights)):
        off = idx * 8
        xs.extend([x_center - dx, x_center + dx, x_center + dx, x_center - dx,
                   x_center - dx, x_center + dx, x_center + dx, x_center - dx])
        ys.extend([yc - dy, yc - dy, yc + dy, yc + dy,
                   yc - dy, yc - dy, yc + dy, yc + dy])
        zs.extend([0.0, 0.0, 0.0, 0.0, float(h), float(h), float(h), float(h)])
        ii.extend([off + v for v in base_i])
        jj.extend([off + v for v in base_j])
        kk.extend([off + v for v in base_k])

    return xs, ys, zs, ii, jj, kk


def update_cuboid_z_only(z_heights: list) -> list:
    zs = []
    for h in z_heights:
        zs.extend([0.0, 0.0, 0.0, 0.0, float(h), float(h), float(h), float(h)])
    return zs


def build_animated_quadrant_3d_bars_and_colormap(
    df_in: pd.DataFrame,
    threshold_pairs: list,
    step_labels: list,
    focus_target: str
) -> go.Figure:
    """
    Animation 1:
    - Panel A (Top-Left, 2D): Class Diagnostic Quadrant (Precision vs. Recall) with F1-Score bubble color,
      light grey full trajectory path, and dashed grey traveled trajectory.
    - Panel B (Top-Right, 3D): 3D Bar Chart across the 3 scales and 5 outcomes with large, crisp HTML/SVG
      3D overlay annotations (`scene.annotations`) that never get culled or hidden.
    - Panel C (Bottom Full-Width, 2D): Flicker-free 100-cell SVG vector grid (`go.Bar` with fixed IDs)
      using 5 smooth continuous row color gradients, high-contrast angled X-ticks (-35 deg) with tick marks,
      and straight Y-ticks (0 deg) with tick marks.
    - Bottom HUD: Live Formula Breakdown positioned cleanly below Panel C.
    """
    m = compute_all_steps_class_and_scale_metrics(df_in, threshold_pairs)
    classes = m["classes"]
    n_cls = len(classes)
    n_steps = len(step_labels)
    gt_counts = m["gt_counts"]
    bubble_sizes = np.clip(np.sqrt(gt_counts) * 5.2, 13, 52)

    if focus_target in classes:
        focus_idx = classes.index(focus_target)
        line_widths = [3.2 if i == focus_idx else 1.1 for i in range(n_cls)]
        line_colors = ["#ffffff" if i == focus_idx else "#1e293b" for i in range(n_cls)]
        bars_3d_all_steps = m["class_3d_bars"][:, focus_idx, :, :]
    else:
        focus_idx = None
        line_widths = [1.2] * n_cls
        line_colors = ["#1e293b"] * n_cls
        bars_3d_all_steps = m["global_3d_bars"]

    c_mat = m["f1_mat"]
    cscale = "Viridis"
    cmin, cmax = 20.0, 100.0
    cbar_title = "F1-Score (%)"

    # 1. Static full-sweep rail lines (light grey)
    rail_x, rail_y = [], []
    for c_i in range(n_cls):
        rail_x.extend(m["rec_mat"][:, c_i].tolist())
        rail_x.append(None)
        rail_y.extend(m["prec_mat"][:, c_i].tolist())
        rail_y.append(None)

    # 2. Fixed-length progressive trail builder (dashed grey, zero jumping)
    def make_fixed_length_trail(step_i: int):
        tx, ty = [], []
        for c_i in range(n_cls):
            r_hist = m["rec_mat"][:step_i + 1, c_i].tolist()
            p_hist = m["prec_mat"][:step_i + 1, c_i].tolist()
            pad_len = n_steps - len(r_hist)
            tx.extend(r_hist + [r_hist[-1]] * pad_len + [None])
            ty.extend(p_hist + [p_hist[-1]] * pad_len + [None])
        return tx, ty

    # Pre-build fixed geometry for the 100 vector cells of the 5x20 table (go.Bar)
    cell_x = []
    cell_base = []
    cell_height = []
    cell_ids = []
    cell_row_names = []
    for r_i in range(5):
        for c_i in range(n_cls):
            cell_x.append(classes[c_i])
            cell_base.append(r_i - 0.44)
            cell_height.append(0.88)
            cell_ids.append(f"cell_{r_i}_{c_i}")
            cell_row_names.append(OUTCOME_NAMES_5[r_i])

    def make_vector_table_colors_and_counts(step_i: int):
        counts_5x20 = np.vstack([
            m["tp_mat"][step_i],
            m["pure_fp_mat"][step_i],
            m["loc_err_mat"][step_i],
            m["mis_mat"][step_i],
            m["conf_drop_fn_mat"][step_i]
        ])
        v_5x20 = m["v_smooth"][step_i]

        hex_colors_100 = []
        text_counts_100 = []
        for r_i in range(5):
            cmap_r = ROW_CMAPS[r_i]
            for c_i in range(n_cls):
                rgba = cmap_r(float(v_5x20[r_i, c_i]))
                hex_colors_100.append(mcolors.to_hex(rgba))
                text_counts_100.append(str(int(counts_5x20[r_i, c_i])))
        return hex_colors_100, text_counts_100

    def make_formula_annotation_html(step_i: int, step_lbl: str) -> str:
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

        return (
            f"<b>Live Formula Breakdown at {step_lbl} — {scope_title}:</b><br>"
            f"• <b>Precision</b> = TP (Conf, IoU&gt;T) / [TP (Conf, IoU&gt;T) + FP (Haluzination) + Missed (IoU&lt;T) + Misclassification] = "
            f"<b>{tp_v}</b> / (<b>{tp_v}</b> + {fp_bg_v} + {loc_v} + {mis_v}) = "
            f"<b>{tp_v} / {tp_v + fp_tot} = {prec_pct:.1f}%</b><br>"
            f"• <b>Recall</b> = TP (Conf, IoU&gt;T) / [TP (Conf, IoU&gt;T) + Missed (Conf&lt;T) + Missed (IoU&lt;T) + Misclassification + Undetected] = "
            f"<b>{tp_v}</b> / (<b>{tp_v}</b> + {fn_conf_v} + {loc_v} + {mis_v} + {fn_blind_v}) = "
            f"<b>{tp_v} / {gt_tot} = {rec_pct:.1f}%</b>"
        )

    max_3d_z = int(max(5, bars_3d_all_steps.max()) * 1.22) + 4
    z_label_offset = max_3d_z * 0.04

    def make_3d_scene_annotations(bars_3x5: np.ndarray) -> list:
        """
        Builds crisp HTML/SVG 3D overlay annotations for:
        1) All 5 Outcome Category labels along the front edge (never culled by WebGL!)
        2) All 3 Scale labels along the right edge
        3) All 15 live bar counts floating above each 3D bar
        """
        anns = []
        # 1. Front X-axis outcome category labels
        short_out_labels = [
            "<b>TP</b><br>(Conf, IoU>T)",
            "<b>FP</b><br>(Haluzination)",
            "<b>Missed</b><br>(IoU<T)",
            "<b>Misclass.</b>",
            "<b>Missed</b><br>(Conf<T)"
        ]
        for out_i, lbl_html in enumerate(short_out_labels):
            anns.append(dict(
                x=float(out_i), y=-0.68, z=0.0,
                text=lbl_html,
                showarrow=False,
                font=dict(size=12, color="#f8fafc"),
                bgcolor="rgba(15, 23, 42, 0.82)",
                bordercolor=OUTCOME_COLORS_5[OUTCOME_NAMES_5[out_i]],
                borderwidth=1.5,
                borderpad=3
            ))

        # 2. Side Y-axis scale labels
        for sc_i, sc_lbl in enumerate(SCALE_SHORT_LABELS):
            anns.append(dict(
                x=4.68, y=float(sc_i), z=0.0,
                text=f"<b>{sc_lbl}</b>",
                showarrow=False,
                font=dict(size=12.5, color="#f8fafc"),
                bgcolor="rgba(30, 41, 59, 0.85)",
                bordercolor="#94a3b8",
                borderwidth=1,
                borderpad=3
            ))

        # 3. Floating count numbers above each of the 15 bars
        for out_i in range(5):
            for sc_i in range(3):
                val = int(bars_3x5[sc_i, out_i])
                anns.append(dict(
                    x=float(out_i),
                    y=float(sc_i),
                    z=float(val) + z_label_offset,
                    text=f"<b>{val}</b>",
                    showarrow=False,
                    font=dict(size=12, color="#ffffff"),
                    bgcolor="rgba(15, 23, 42, 0.72)",
                    borderpad=2
                ))
        return anns

    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.58, 0.42],
        column_widths=[0.50, 0.50],
        specs=[
            [{"type": "xy"}, {"type": "scene"}],
            [{"type": "xy", "colspan": 2}, None]
        ],
        subplot_titles=(
            "A. 2D Class Diagnostic Quadrant (Precision vs. Recall Trajectories)",
            f"B. 3D Scale & Outcome Bars ({'Class: ' + focus_target if focus_idx is not None else 'All 20 Classes'})",
            "C. Per-Class Dynamic Outcome Intensity Map (5 Outcomes × 20 VOC Classes)"
        ),
        vertical_spacing=0.13,
        horizontal_spacing=0.07
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=14, color="#f8fafc")
    base_annotations = list(fig.layout.annotations)

    # Trace 0: Static full trajectory path (light grey)
    fig.add_trace(
        go.Scatter(
            x=rail_x, y=rail_y,
            mode="lines",
            name="Full Trajectory Path",
            line=dict(color="rgba(203, 213, 225, 0.55)", width=1.5, dash="dot"),
            hoverinfo="skip",
            showlegend=True
        ),
        row=1, col=1
    )

    # Trace 1: Progressive traveled trajectory (dashed grey)
    tx0, ty0 = make_fixed_length_trail(0)
    fig.add_trace(
        go.Scatter(
            x=tx0, y=ty0,
            mode="lines",
            name="Traveled Trajectory",
            line=dict(color="rgba(148, 163, 184, 0.95)", width=2.4, dash="dash"),
            hoverinfo="skip",
            showlegend=True
        ),
        row=1, col=1
    )

    # Trace 2: The 20 Class Bubbles (colored by F1-Score %)
    def make_customdata(step_i: int):
        return np.stack([
            classes,
            gt_counts,
            m["tp_mat"][step_i],
            m["pure_fp_mat"][step_i],
            m["loc_err_mat"][step_i],
            m["mis_mat"][step_i],
            m["conf_drop_fn_mat"][step_i],
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
            text=classes,
            textposition="top right",
            textfont=dict(size=11, color="#f8fafc"),
            marker=dict(
                size=bubble_sizes,
                color=c_mat[0],
                colorscale=cscale,
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                colorbar=dict(
                    title=dict(text=cbar_title, font=dict(color="#f8fafc", size=12)),
                    tickfont=dict(color="#f8fafc", size=11),
                    len=0.45, x=0.475, y=0.78, thickness=12
                ),
                line=dict(width=line_widths, color=line_colors),
                opacity=0.92
            ),
            customdata=make_customdata(0),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (Total GT: %{customdata[1]})<br>"
                "<b>Precision: %{y:.1f}%</b> | <b>Recall: %{x:.1f}%</b> | <b>F1: %{customdata[9]:.1f}%</b><br>"
                "  ├─ TP (Conf, IoU>T): %{customdata[2]}<br>"
                "  ├─ FP (Haluzination): %{customdata[3]}<br>"
                "  ├─ Missed (IoU<T): %{customdata[4]}<br>"
                "  ├─ Misclassification: %{customdata[5]}<br>"
                "  └─ Missed (Conf<T): %{customdata[6]}<extra></extra>"
            )
        ),
        row=1, col=1
    )

    # Traces 3..7: 5 3D Bar Mesh traces (one per Outcome Category across the 3 Scales)
    y_scale_centers = [0.0, 1.0, 2.0]
    bars0_3x5 = bars_3d_all_steps[0]
    for out_i, out_name in enumerate(OUTCOME_NAMES_5):
        xs_m, ys_m, zs_m, ii_m, jj_m, kk_m = build_cuboid_mesh_arrays(
            x_center=float(out_i),
            y_centers=y_scale_centers,
            z_heights=bars0_3x5[:, out_i].tolist(),
            dx=0.26,
            dy=0.26
        )
        fig.add_trace(
            go.Mesh3d(
                x=xs_m, y=ys_m, z=zs_m,
                i=ii_m, j=jj_m, k=kk_m,
                name=out_name,
                color=OUTCOME_COLORS_5[out_name],
                opacity=0.94,
                flatshading=True,
                lighting=dict(ambient=0.72, diffuse=0.82, roughness=0.35, specular=0.18),
                hoverinfo="name+z",
                showlegend=True
            ),
            row=1, col=2
        )

    # Trace 8: Flicker-Free 5 x 20 SVG Vector Grid (go.Bar with 100 persistent cell IDs)
    hex0_100, txt0_100 = make_vector_table_colors_and_counts(0)
    fig.add_trace(
        go.Bar(
            x=cell_x,
            y=cell_height,
            base=cell_base,
            ids=cell_ids,
            width=0.92,
            name="5×20 Outcome Grid",
            marker=dict(
                color=hex0_100,
                line=dict(width=1.2, color="#1e293b")
            ),
            text=txt0_100,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=12, color="#0f172a"),
            customdata=np.stack([cell_row_names, txt0_100], axis=-1),
            hovertemplate="Class: <b>%{x}</b><br>Metric: <b>%{customdata[0]}</b><br>Count: <b>%{customdata[1]}</b><extra></extra>",
            showlegend=False
        ),
        row=2, col=1
    )

    # Reference lines in Panel A
    fig.add_shape(
        type="line", x0=80, x1=80, y0=0, y1=105,
        line=dict(color="rgba(148,163,184,0.5)", width=1.5, dash="dash"),
        xref="x1", yref="y1"
    )
    fig.add_shape(
        type="line", x0=0, x1=105, y0=70, y1=70,
        line=dict(color="rgba(148,163,184,0.5)", width=1.5, dash="dash"),
        xref="x1", yref="y1"
    )

    def make_frame_annotations(formula_html_str: str):
        live_box = dict(
            text=formula_html_str,
            xref="paper", yref="paper",
            x=0.5, y=-0.12,
            xanchor="center", yanchor="top",
            showarrow=False,
            align="left",
            font=dict(size=13, color="#0f172a"),
            bgcolor="rgba(241, 245, 249, 0.97)",
            bordercolor="#94a3b8",
            borderwidth=1.5,
            borderpad=9
        )
        return base_annotations + [live_box]

    frames = []
    for s_idx, lbl in enumerate(step_labels):
        tx_s, ty_s = make_fixed_length_trail(s_idx)
        bars_s_3x5 = bars_3d_all_steps[s_idx]
        hex_s_100, txt_s_100 = make_vector_table_colors_and_counts(s_idx)
        form_s = make_formula_annotation_html(s_idx, lbl)

        frame_data = [
            # Trace 1: Traveled Trajectory
            go.Scatter(x=tx_s, y=ty_s),
            # Trace 2: Class Bubbles
            go.Scatter(
                x=m["rec_mat"][s_idx],
                y=m["prec_mat"][s_idx],
                ids=classes,
                text=classes,
                marker=dict(
                    size=bubble_sizes,
                    color=c_mat[s_idx],
                    colorscale=cscale,
                    cmin=cmin,
                    cmax=cmax,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=cbar_title, font=dict(color="#f8fafc", size=12)),
                        tickfont=dict(color="#f8fafc", size=11),
                        len=0.45, x=0.475, y=0.78, thickness=12
                    ),
                    line=dict(width=line_widths, color=line_colors),
                    opacity=0.92
                ),
                customdata=make_customdata(s_idx)
            ),
        ]

        # Traces 3..7: Update z vertices for each of the 5 3D Outcome Bar Meshes
        for out_i in range(5):
            frame_data.append(
                go.Mesh3d(z=update_cuboid_z_only(bars_s_3x5[:, out_i].tolist()))
            )

        # Trace 8: Update 5x20 SVG Vector Grid colors and counts
        frame_data.append(
            go.Bar(
                marker=dict(color=hex_s_100, line=dict(width=1.2, color="#1e293b")),
                text=txt_s_100,
                customdata=np.stack([cell_row_names, txt_s_100], axis=-1)
            )
        )

        frames.append(go.Frame(
            name=lbl,
            data=frame_data,
            traces=[1, 2, 3, 4, 5, 6, 7, 8],
            layout=go.Layout(
                annotations=make_frame_annotations(form_s),
                scene=dict(annotations=make_3d_scene_annotations(bars_s_3x5))
            )
        ))

    fig.frames = frames

    sliders = [{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {"prefix": "Active Step: ", "visible": True, "xanchor": "right", "font": {"size": 13, "color": "#f8fafc"}},
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 170},
        "len": 0.86,
        "x": 0.14,
        "y": 0,
        "font": {"color": "#cbd5e1"},
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
        height=1020,
        barmode="overlay",
        annotations=make_frame_annotations(make_formula_annotation_html(0, step_labels[0])),
        xaxis1=dict(
            range=[-3, 105],
            title=dict(text="True Positive Recall Rate (%)", font=dict(size=13, color="#f8fafc")),
            tickfont=dict(size=11.5, color="#e2e8f0")
        ),
        yaxis1=dict(
            range=[5, 105],
            title=dict(text="Precision (%)", font=dict(size=13, color="#f8fafc")),
            tickfont=dict(size=11.5, color="#e2e8f0")
        ),
        scene=dict(
            xaxis=dict(
                title="",
                range=[-0.6, 4.6],
                showticklabels=False,
                gridcolor="rgba(148, 163, 184, 0.25)",
                backgroundcolor="rgba(15, 23, 42, 0.35)",
                showbackground=True
            ),
            yaxis=dict(
                title="",
                range=[-0.6, 2.6],
                showticklabels=False,
                gridcolor="rgba(148, 163, 184, 0.25)",
                backgroundcolor="rgba(15, 23, 42, 0.35)",
                showbackground=True
            ),
            zaxis=dict(
                title=dict(text="Instances", font=dict(size=13, color="#f8fafc")),
                range=[0, max_3d_z],
                tickfont=dict(size=12, color="#f8fafc"),
                gridcolor="rgba(148, 163, 184, 0.30)",
                backgroundcolor="rgba(15, 23, 42, 0.45)",
                showbackground=True
            ),
            annotations=make_3d_scene_annotations(bars0_3x5),
            aspectratio=dict(x=1.55, y=1.12, z=0.92),
            camera=dict(
                eye=dict(x=1.52, y=-1.65, z=1.15),
                center=dict(x=0.05, y=-0.05, z=-0.06)
            )
        ),
        xaxis2=dict(
            title="",
            type="category",
            categoryorder="array",
            categoryarray=classes,
            tickangle=-35,
            ticks="outside",
            ticklen=7,
            tickwidth=1.8,
            tickcolor="#94a3b8",
            showline=True,
            linecolor="#94a3b8",
            linewidth=1.5,
            tickfont=dict(size=12, color="#f8fafc"),
            showgrid=False
        ),
        yaxis2=dict(
            title="",
            range=[4.55, -0.55],
            tickvals=[0, 1, 2, 3, 4],
            ticktext=OUTCOME_NAMES_5,
            tickangle=0,
            ticks="outside",
            ticklen=7,
            tickwidth=1.8,
            tickcolor="#94a3b8",
            showline=True,
            linecolor="#94a3b8",
            linewidth=1.5,
            tickfont=dict(size=12.5, color="#f8fafc"),
            showgrid=False,
            zeroline=False
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5,
            font=dict(size=11.5, color="#f8fafc")
        ),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0,
            "y": 0.0,
            "xanchor": "left",
            "yanchor": "top",
            "pad": {"t": 170, "r": 10},
            "font": {"color": "#0f172a"},
            "bgcolor": "#e2e8f0",
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 110, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                }
            ]
        }],
        sliders=sliders,
        margin=dict(l=180, r=30, t=90, b=255)
    )
    return fig


def build_animated_phase_space_and_scale_bars(
    df_in: pd.DataFrame,
    threshold_pairs: list,
    step_labels: list
) -> go.Figure:
    """
    Animation 2:
    Synchronized two-panel animation with unified naming and colors matching Animation 1:
    - 'TP (Conf, IoU>T)' (Green)
    - 'FP (Haluzination)' (Orange)
    - 'Missed (IoU<T)' (Blue)
    - 'Misclassification' (Red)
    - 'Missed (Conf<T)' (Grey)
    """
    base_d = evaluate_at_thresholds_vectorized(df_in, threshold_pairs[0][0], threshold_pairs[0][1])
    plot_base = base_d[base_d["Candidate_Confidence"].notna() & (~base_d["Is_Duplicate_Candidate_Row"])].copy()

    x_fixed = plot_base["Candidate_Confidence"].to_numpy()
    y_fixed = plot_base["IoU"].to_numpy()
    rec_ids = plot_base["Record_ID"].tolist()
    symbols_fixed = [SCALE_SYMBOLS.get(s, "circle") for s in plot_base["Environmental_Condition"]]

    scale_base = df_in[df_in["Candidate_Confidence"].notna() & (~df_in["Is_Duplicate_Candidate_Row"])].copy()
    active_scales = [s for s in SCALE_ORDER if s in scale_base["Environmental_Condition"].unique()]
    outcome_order = [
        "TP (Conf, IoU>T)",
        "Misclassification",
        "Missed (IoU<T)",
        "Missed (Conf<T)",
        "FP (Haluzination)"
    ]

    def compute_step_data(conf_t: float, iou_t: float):
        d_scatter = evaluate_at_thresholds_vectorized(plot_base, conf_t, iou_t)
        states = d_scatter["Phase_State"].tolist()
        colors = [OUTCOME_COLORS_5.get(st_name, "#cbd5e1") for st_name in states]
        opacities = [0.24 if st_name == "Missed (Conf<T)" else 0.88 for st_name in states]
        sizes = [6.5 if st_name == "Missed (Conf<T)" else 9.0 for st_name in states]
        custom = np.stack([
            d_scatter["Sample_ID"],
            d_scatter["Ground_Truth"],
            d_scatter["Effective_Pred_Class"],
            d_scatter["Environmental_Condition"],
            d_scatter["Phase_State"]
        ], axis=-1)

        d_scale = evaluate_at_thresholds_vectorized(scale_base, conf_t, iou_t)
        d_scale_active = d_scale[d_scale["Dynamic_Outcome"].isin(outcome_order)]

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

    for ann in fig.layout.annotations:
        ann.font = dict(size=14, color="#f8fafc")

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

    for out in outcome_order:
        fig.add_trace(
            go.Bar(
                x=active_scales,
                y=pcts0[out],
                name=out,
                marker=dict(color=OUTCOME_COLORS_5[out]),
                text=txts0[out],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="white", size=11),
                hovertemplate=f"<b>%{{x}}</b><br>{out}: <b>%{{y:.1f}}%</b><extra></extra>"
            ),
            row=1, col=2
        )

    def make_gate_shapes(conf_t: float, iou_t: float):
        return [
            dict(
                type="rect", x0=conf_t, x1=1.02, y0=iou_t, y1=1.03,
                fillcolor="rgba(44, 160, 44, 0.08)", line=dict(width=0),
                layer="below", xref="x1", yref="y1"
            ),
            dict(
                type="rect", x0=-0.03, x1=conf_t, y0=-0.04, y1=1.03,
                fillcolor="rgba(148, 163, 184, 0.10)", line=dict(width=0),
                layer="below", xref="x1", yref="y1"
            ),
            dict(
                type="line", x0=conf_t, x1=conf_t, y0=-0.04, y1=1.03,
                line=dict(color="#2563eb", width=2.5, dash="dot"),
                xref="x1", yref="y1"
            ),
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
                    marker=dict(color=OUTCOME_COLORS_5[out])
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
        "currentvalue": {"prefix": "Active Gate: ", "visible": True, "xanchor": "right", "font": {"size": 13, "color": "#f8fafc"}},
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 45},
        "len": 0.86,
        "x": 0.14,
        "y": 0,
        "font": {"color": "#cbd5e1"},
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
        xaxis1=dict(
            range=[-0.03, 1.02],
            title=dict(text="Prediction / Candidate Confidence Score", font=dict(size=13, color="#f8fafc")),
            tickfont=dict(size=11.5, color="#e2e8f0")
        ),
        yaxis1=dict(
            range=[-0.04, 1.04],
            title=dict(text="Intersection over Union (IoU with GT)", font=dict(size=13, color="#f8fafc")),
            tickfont=dict(size=11.5, color="#e2e8f0")
        ),
        xaxis2=dict(
            title=dict(text="Object Scale Category (% of Image Area)", font=dict(size=13, color="#f8fafc")),
            tickfont=dict(size=11.5, color="#e2e8f0")
        ),
        yaxis2=dict(
            range=[0, 105],
            title=dict(text="Percentage of Records (%)", font=dict(size=13, color="#f8fafc")),
            tickfont=dict(size=11.5, color="#e2e8f0")
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5,
            font=dict(size=11.5, color="#f8fafc")
        ),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0,
            "y": 0.0,
            "xanchor": "left",
            "yanchor": "top",
            "pad": {"t": 45, "r": 10},
            "font": {"color": "#0f172a"},
            "bgcolor": "#e2e8f0",
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 110, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                }
            ]
        }],
        sliders=sliders,
        margin=dict(l=60, r=30, t=95, b=95)
    )
    return fig


eval_df = load_evaluation_data()

# ==============================================================================
# Header & Sidebar Parameters
# ==============================================================================
st.title("YOLO26n Behavioral & Visual Analytics Studio")
st.caption(
    "Explainable Object Detection on Pascal VOC: Smooth Multi-Panel Threshold "
    "Animations with 3D Scale Bars, 5×20 Outcome Intensity Maps, and Phase Space."
)

conf_threshold = 0.25
iou_threshold = 0.50

# ==============================================================================
# Animated Diagnostic Studio (Trajectories, 3D Bars, 5x20 Map & Phase Space)
# ==============================================================================
st.subheader("Animated Diagnostic Studio: 2D Trajectories, 3D Scale Bars, 5×20 Intensity Map & Phase Space")

with st.expander("ℹ️ Click to explain: Precision, Recall, IoU & All 5 Outcome Categories (With Simple Examples)", expanded=False):
    st.markdown(
        """
        ### 1. Core Metrics Explained Simply
        * **Intersection over Union ($\text{IoU}$):**  
          Measures how accurately a predicted box overlaps the true object box ($0.0$ = no overlap at all, $1.0$ = pixel-perfect match).
        * **Precision ($\text{When the model predicts an object, how often is it really an accurate hit?}$):**  
          $$\\text{Precision} = \\frac{\\text{TP (Conf, IoU>T)}}{\\text{TP (Conf, IoU>T)} + \\text{FP (Haluzination)} + \\text{Missed (IoU<T)} + \\text{Misclassification}}$$
        * **Recall ($\text{Out of all real objects in the dataset, what percentage did the model find?}$):**  
          $$\\text{Recall} = \\frac{\\text{TP (Conf, IoU>T)}}{\\text{Total Ground Truth}}$$

        ---
        ### 2. The 5 Dynamic Categories Across Animation 1 & Animation 2
        * 🟢 **TP (Conf, IoU>T):** Correct class label, $\\text{Confidence} \\ge T_{\\text{conf}}$, and $\\text{IoU} \\ge T_{\\text{IoU}}$. Drives both Precision and Recall up.
        * 🟠 **FP (Haluzination):** The model predicted an object on background or produced a redundant box ($\\text{Conf} \\ge T_{\\text{conf}}$). Pulls Precision down.
        * 🔵 **Missed (IoU<T):** The model detected the object ($\\text{Conf} \\ge T_{\\text{conf}}$), but the bounding box overlap is below $T_{\\text{IoU}}$. Pulls **both** Precision and Recall down.
        * 🔴 **Misclassification:** Good bounding box overlap ($\\text{IoU} \\ge T_{\\text{IoU}}$), but wrong class label (e.g., predicting `cat` on a `dog`).
        * ⚪ **Missed (Conf<T):** Candidate box confidence fell below $T_{\\text{conf}}$, so the detection was filtered out (greyed out). Pulls Recall down.
        """
    )

if eval_df.empty:
    st.error("No `voc_behavioral_results.csv` found! Please run `python offline_test.py` first.")
else:
    c_top1, c_top2 = st.columns([1.6, 1.4])
    with c_top1:
        sweep_mode = st.radio(
            "Select Parameter Sweep to Animate:",
            [
                f"Confidence Threshold Sweep (0.25 → 0.90 at fixed IoU = {iou_threshold:.2f})",
                f"IoU Match Requirement Sweep (0.15 → 0.90 at fixed Conf = {conf_threshold:.2f})",
                "Joint Strictness Sweep (Simultaneous Conf 0.25→0.85 & IoU 0.25→0.85)"
            ]
        )
    with c_top2:
        focus_target = st.selectbox(
            "Inspect Scope for 3D Bar Chart & Live Formula:",
            ["All 20 Classes (Dataset Total)"] + VOC_CLASSES_ALPHABETICAL,
            index=0,
            help="Select 'All 20 Classes' or pick a specific class (e.g., 'dog', 'chair', 'person') to watch its 3D scale bars and live formula numbers update!"
        )

    # Always use the finest 0.01 resolution and all object scales
    step_size = 0.01
    sub_anim_df = eval_df.copy()

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
    # ANIMATION 1: 2D Quadrant + 3D Scale Bars + 5x20 Color Map + Live Formula
    # ======================================================================
    st.markdown("---")
    st.markdown("### Animation 1: 2D Class Diagnostic Quadrant, 3D Scale Bars & 5×20 Per-Class Intensity Map")

    fig_quad_anim = build_animated_quadrant_3d_bars_and_colormap(
        sub_anim_df,
        threshold_pairs=threshold_pairs,
        step_labels=step_labels,
        focus_target=focus_target
    )
    st.plotly_chart(fig_quad_anim, use_container_width=True)

    # ======================================================================
    # ANIMATION 2: Synchronized Phase Space (Panel A) & Scale Bars (Panel B)
    # ======================================================================
    st.markdown("---")
    st.markdown("### Animation 2: Synchronized Joint Phase Space (Panel A) & Scale Sensitivity Proportions (Panel B)")

    fig_phase_scale_anim = build_animated_phase_space_and_scale_bars(
        sub_anim_df,
        threshold_pairs=threshold_pairs,
        step_labels=step_labels
    )
    st.plotly_chart(fig_phase_scale_anim, use_container_width=True)