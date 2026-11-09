#!/usr/bin/env python3
"""
Standalone-Skript zur Erstellung der 2D Class Diagnostic Quadrant Animationen
(Precision vs. Recall Trajectories) für LibreOffice Impress.

Erzeugt:
1. quadrant_confidence_sweep.mp4 (sowie .webm / .gif Fallback)
2. quadrant_iou_sweep.mp4 (sowie .webm / .gif Fallback)
"""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# =============================================================================
# 1. KONFIGURATION & DATEN LADEN
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CSV_PATH = BASE_DIR / "voc_behavioral_results.csv"

if not CSV_PATH.exists():
    alt_path = Path(r"C:\Users\Simon Mader\Documents\AI\3. Semester\AI_and_visualization\AI_visualization\Andres-Simon\phase1\voc_behavioral_results.csv")
    if alt_path.exists():
        CSV_PATH = alt_path

VOC_CLASSES_SEMANTIC = [
    # Animals & Person
    "bird", "cat", "cow", "dog", "horse", "sheep", "person",
    # Vehicles
    "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
    # Indoor Objects
    "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
]
VOC_CLASSES_ALPHABETICAL = sorted(VOC_CLASSES_SEMANTIC)


def load_evaluation_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV-Datei nicht gefunden unter: {csv_path}\n"
            "Bitte stelle sicher, dass 'voc_behavioral_results.csv' im selben Ordner liegt."
        )

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


# =============================================================================
# 2. METRIKEN FÜR ALLE SWEEP-SCHRITTE VEKTORISIERT BERECHNEN
# =============================================================================
def compute_quadrant_metrics(df_in: pd.DataFrame, threshold_pairs: list) -> dict:
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

    rec_mat = np.zeros((n_steps, n_cls))
    prec_mat = np.zeros((n_steps, n_cls))
    f1_mat = np.zeros((n_steps, n_cls))

    valid_cls = eff_cls != "None"
    same_cls = eff_cls == gt_cls

    for s_idx, (c_t, i_t) in enumerate(threshold_pairs):
        pass_conf = conf >= c_t
        pass_iou = iou >= i_t

        gt_active = is_gt & has_box & pass_conf
        tp_mask = gt_active & pass_iou & valid_cls & same_cls
        mis_mask = gt_active & pass_iou & valid_cls & (~same_cls)
        loc_mask = gt_active & (~(pass_iou & valid_cls))
        fp_active_mask = is_pure_fp_row & pass_conf

        tp_c = (gt_cls_masks & tp_mask).sum(axis=1)
        mis_c = (gt_cls_masks & mis_mask).sum(axis=1)
        loc_c = (eff_cls_gt_masks & loc_mask).sum(axis=1)
        fp_c = (fp_cls_masks & fp_active_mask).sum(axis=1)

        total_fp_c = fp_c + loc_c + mis_c
        denom_p = tp_c + total_fp_c

        rec_c = np.where(gt_counts > 0, (tp_c / np.maximum(1, gt_counts)) * 100.0, 0.0)
        prec_c = np.where(denom_p > 0, (tp_c / np.maximum(1, denom_p)) * 100.0, np.where(tp_c == 0, 100.0, 0.0))
        f1_c = np.where((prec_c + rec_c) > 0, 2 * prec_c * rec_c / np.maximum(1e-6, prec_c + rec_c), 0.0)

        rec_mat[s_idx] = np.round(rec_c, 1)
        prec_mat[s_idx] = np.round(prec_c, 1)
        f1_mat[s_idx] = np.round(f1_c, 1)

    return {
        "classes": cls_list,
        "gt_counts": gt_counts,
        "rec_mat": rec_mat,
        "prec_mat": prec_mat,
        "f1_mat": f1_mat,
    }


# =============================================================================
# 3. ANIMATION ERSTELLEN & SPEICHERN (INKL. LIVE HUD & END-PAUSE)
# =============================================================================
def create_and_save_quadrant_animation(
    df_in: pd.DataFrame,
    threshold_pairs: list,
    sweep_title: str,
    active_mode: str,
    output_stem: str,
    fps: int = 4,
    hold_seconds: float = 2.5
):
    m = compute_quadrant_metrics(df_in, threshold_pairs)
    classes = m["classes"]
    n_cls = len(classes)
    n_steps = len(threshold_pairs)
    gt_counts = m["gt_counts"]

    # Standzeit-Frames am Ende berechnen
    hold_frames = int(round(fps * hold_seconds))
    total_frames = n_steps + hold_frames

    plotly_diameters = np.clip(np.sqrt(gt_counts) * 5.2, 13, 52)
    mpl_sizes = (plotly_diameters * 1.15) ** 2

    bg_color = "#13171f"
    grid_color = "#262c36"
    text_color = "#e2e8f0"
    title_color = "#f8fafc"

    fig, ax = plt.subplots(figsize=(14.6, 6.2), dpi=150)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-3, 105)
    ax.set_ylim(5, 105)
    ax.set_xlabel("True Positive Recall Rate (%)", fontsize=12.5, color=title_color, labelpad=12)
    ax.set_ylabel("Precision (%)", fontsize=12.5, color=title_color, labelpad=12)
    ax.set_title(
        sweep_title,
        fontsize=13.5, fontweight="medium", color=title_color, pad=16
    )

    ax.tick_params(axis="both", colors=text_color, labelsize=10.5, length=0)
    ax.yaxis.grid(True, color=grid_color, linestyle="-", linewidth=1.0, alpha=0.9)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.axvline(80, color="#94a3b8", linestyle=(0, (5, 5)), linewidth=1.4, alpha=0.55, zorder=2)
    ax.axhline(70, color="#94a3b8", linestyle=(0, (5, 5)), linewidth=1.4, alpha=0.55, zorder=2)

    # 1. Statische Gesamttrajektorie (Hintergrund)
    for c_i in range(n_cls):
        ax.plot(
            m["rec_mat"][:, c_i],
            m["prec_mat"][:, c_i],
            color="#cbd5e1",
            linestyle=":",
            linewidth=1.2,
            alpha=0.42,
            zorder=3
        )

    # 2. Dynamische bereits zurückgelegte Trajektorien
    traveled_lines = []
    for c_i in range(n_cls):
        (line,) = ax.plot(
            [m["rec_mat"][0, c_i]],
            [m["prec_mat"][0, c_i]],
            color="#94a3b8",
            linestyle="--",
            linewidth=2.0,
            alpha=0.92,
            zorder=4
        )
        traveled_lines.append(line)

    # 3. Klassensymbole (Bubbles)
    norm = Normalize(vmin=20.0, vmax=100.0)
    cmap = plt.get_cmap("viridis")

    scatter = ax.scatter(
        m["rec_mat"][0],
        m["prec_mat"][0],
        s=mpl_sizes,
        c=m["f1_mat"][0],
        cmap=cmap,
        norm=norm,
        alpha=0.90,
        edgecolors="#1e293b",
        linewidths=1.4,
        zorder=5
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.03)
    cbar.outline.set_edgecolor("#334155")
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(colors=text_color, labelsize=10, length=0)
    cbar.ax.set_title("F1-Score (%)", color=title_color, fontsize=11, pad=10, loc="left")

    hud_box = ax.text(
        0.025, 0.07,
        "",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color="#f8fafc",
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.65",
            facecolor="#1e293b",
            edgecolor="#3b82f6" if active_mode == "conf" else "#f59e0b",
            linewidth=1.8,
            alpha=0.94
        ),
        zorder=10
    )

    def format_hud_text(c_val: float, i_val: float, step_i: int) -> str:
        mean_f1 = m["f1_mat"][step_i].mean()
        status_suffix = " (FINAL)" if step_i >= n_steps - 1 else ""
        if active_mode == "conf":
            return (
                f"▶ ACTIVE SWEEP: Confidence Threshold{status_suffix}\n"
                f"   Confidence ≥ {c_val:.2f}   |   IoU ≥ {i_val:.2f} (fixed)\n"
                f"   Mean F1: {mean_f1:.1f}%"
            )
        else:
            return (
                f"▶ ACTIVE SWEEP: IoU Requirement{status_suffix}\n"
                f"   IoU ≥ {i_val:.2f}   |   Confidence ≥ {c_val:.2f} (fixed)\n"
                f"   Mean F1: {mean_f1:.1f}%"
            )

    hud_box.set_text(format_hud_text(threshold_pairs[0][0], threshold_pairs[0][1], 0))

    plt.tight_layout()

    def update(frame_idx: int):
        # Wenn im Hold-Bereich am Ende: beim letzten Index einfrieren
        data_idx = min(frame_idx, n_steps - 1)

        # Update zurückgelegte Trajektorien
        for c_i, line in enumerate(traveled_lines):
            line.set_data(
                m["rec_mat"][:data_idx + 1, c_i],
                m["prec_mat"][:data_idx + 1, c_i]
            )

        # Update Bubbles
        coords = np.column_stack((m["rec_mat"][data_idx], m["prec_mat"][data_idx]))
        scatter.set_offsets(coords)
        scatter.set_array(m["f1_mat"][data_idx])

        # Update HUD-Text
        c_val, i_val = threshold_pairs[data_idx]
        hud_box.set_text(format_hud_text(c_val, i_val, data_idx))

        return [*traveled_lines, scatter, hud_box]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 // fps,
        blit=False  # Wichtig: blit=False garantiert sauberen Export aller Frames ohne Pipeline-Drops
    )

    has_ffmpeg = shutil.which("ffmpeg") is not None

    if has_ffmpeg:
        # MP4 (H.264)
        mp4_path = BASE_DIR / f"{output_stem}.mp4"
        print(f"Speichere MP4-Video ({total_frames} Frames): {mp4_path.name} ...")
        writer_mp4 = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=[
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"
            ]
        )
        ani.save(str(mp4_path), writer=writer_mp4)
        print(f"✔ Gespeichert: {mp4_path}")

        # WebM (VP9)
        webm_path = BASE_DIR / f"{output_stem}.webm"
        print(f"Speichere WebM-Video: {webm_path.name} ...")
        writer_webm = animation.FFMpegWriter(
            fps=fps,
            codec="libvpx-vp9",
            extra_args=[
                "-b:v", "2M",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"
            ]
        )
        ani.save(str(webm_path), writer=writer_webm)
        print(f"✔ Gespeichert: {webm_path}")
    else:
        gif_path = BASE_DIR / f"{output_stem}.gif"
        writer_gif = animation.PillowWriter(fps=fps)
        ani.save(str(gif_path), writer=writer_gif)
        print(f"✔ Gespeichert: {gif_path}")

    plt.close(fig)


# =============================================================================
# 4. HAUPTPROGRAMM: CONFIDENCE-SWEEP & IOU-SWEEP GENERIEREN
# =============================================================================
if __name__ == "__main__":
    print("Lade Evaluationsdaten...")
    eval_df = load_evaluation_data(CSV_PATH)

    fixed_conf = 0.25
    fixed_iou = 0.50

    # 1. CONFIDENCE THRESHOLD SWEEP (0.25 -> 0.90 inkl. Endwert)
    # linspace stellt sicher, dass exakt 0.25 und 0.90 inklusive sind (66 Schritte bei 0.01)
    num_conf_steps = int(round((0.90 - 0.25) / 0.01)) + 1
    c_steps = np.round(np.linspace(0.25, 0.90, num_conf_steps), 2)
    conf_pairs = [(float(c), float(fixed_iou)) for c in c_steps]

    create_and_save_quadrant_animation(
        df_in=eval_df,
        threshold_pairs=conf_pairs,
        sweep_title="A. 2D Class Diagnostic Quadrant (Precision vs. Recall Trajectories) — Confidence Sweep",
        active_mode="conf",
        output_stem="quadrant_confidence_sweep",
        fps=6,             # Leicht erhöht auf 6 fps für flüssigeren Ablauf
        hold_seconds=3.0   # Bleibt am Ende 3 Sekunden lang auf dem finalen Frame stehen
    )

    # 2. IOU REQUIREMENT SWEEP (0.15 -> 0.90 inkl. Endwert)
    num_iou_steps = int(round((0.90 - 0.15) / 0.01)) + 1
    i_steps = np.round(np.linspace(0.15, 0.90, num_iou_steps), 2)
    iou_pairs = [(float(fixed_conf), float(i)) for i in i_steps]

    create_and_save_quadrant_animation(
        df_in=eval_df,
        threshold_pairs=iou_pairs,
        sweep_title="A. 2D Class Diagnostic Quadrant (Precision vs. Recall Trajectories) — IoU Requirement Sweep",
        active_mode="iou",
        output_stem="quadrant_iou_sweep",
        fps=16,
        hold_seconds=3.0   # Bleibt am Ende ebenfalls 3 Sekunden lang stehen
    )

    print("\nFertig! Die Animationen bleiben nun am Ende jeweils 3 Sekunden im finalen Zustand stehen.")