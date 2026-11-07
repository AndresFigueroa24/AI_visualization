from pathlib import Path
import base64
import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.graph_objects as go
from PIL import Image, ImageDraw

# Configuration
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "phase1_results.csv"
IMAGE_DIR = ROOT / "VOC2007" / "JPEGImages"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

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
VOC_ORDER = [cls for classes in CLASS_TYPES.values() for cls in classes]
STATUSES = ["correct", "wrong_class", "missed"]
COLORS = {
    "correct": "#2ca02c",
    "wrong_class": "#ff7f0e",
    "missed": "#d62728",
}
LABELS = {
    "correct": "Correct",
    "wrong_class": "Mislabelled",
    "missed": "Missed",
}
SIZES = ["small", "medium", "large"]

# Ground-truth object-level results from YOLO evaluation.
df = pd.read_csv(CSV_PATH, dtype={"image": str})
df["image"] = df["image"].str.strip().str.zfill(6)
df["class_type"] = df["cls"].map(CLASS_TO_TYPE)
df = df.reset_index(drop=True)
df["hover_id"] = df.index.astype(str)

sns.set_theme(style="white")

# For saving plots with consistent layout and resolution.
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", OUT / filename)

# 1. Missed detection rate by object size
summary = (
    df.groupby("size_bucket", observed=True)["status"]
    .agg(total="size", missed=lambda s: (s == "missed").sum())
    .reindex(SIZES, fill_value=0)
)
summary["rate"] = (100 * summary["missed"] / summary["total"].replace(0, np.nan))
fig2, ax = plt.subplots(figsize=(8.5, 5))
bars = ax.bar([s.title() for s in SIZES], summary["rate"].fillna(0),
              color=["#e2a14a", "#9ab6a5", "#578b7c"], width=0.58)
for bar, (_, row) in zip(bars, summary.iterrows()):
    label = (f"{row['rate']:.1f}%\n({int(row['missed'])}/{int(row['total'])})"
             if row["total"] else "No data")
    ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6), textcoords="offset points", ha="center", va="bottom")
max_rate = summary["rate"].max(skipna=True)
ax.set_ylim(0, max(10, float(max_rate) * 1.3 if pd.notna(max_rate) else 10))
ax.set_xlabel("Ground-truth object size")
ax.set_ylabel("Missed detection rate")
ax.set_title("Missed Detection Rate by Object Size")
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
save_plot("1_missed_rate_by_size.png")

# 2. Object-level confusion matrix with jittered points
plot = df[df["cls"].isin(VOC_ORDER)].copy()
plot["pred_display"] = np.where(plot["status"] == "missed", "Missed", plot["det_cls"])
columns = VOC_ORDER + ["Missed"]
plot = plot[plot["pred_display"].isin(columns)].copy()
plot["x_idx"] = pd.Categorical(plot["pred_display"], categories=columns).codes
plot["y_idx"] = pd.Categorical(plot["cls"], categories=VOC_ORDER).codes
rng = np.random.default_rng(42)
plot["x_dot"] = plot["x_idx"] + rng.uniform(-0.44, 0.44, len(plot))
plot["y_dot"] = plot["y_idx"] + rng.uniform(-0.44, 0.44, len(plot))

fig3, ax = plt.subplots(figsize=(17, 13))
start = 0
for class_type, classes in CLASS_TYPES.items():
    n = len(classes)
    ax.add_patch(Rectangle((start - 0.5, start - 0.5), n, n,
                           fill=False, edgecolor="#d36b6b", linewidth=1.7, zorder=3))
    start += n

sns.scatterplot(
    data=plot, x="x_dot", y="y_dot", hue="status",
    hue_order=STATUSES, palette=COLORS, s=12, alpha=0.75,
    edgecolor="none", ax=ax, zorder=2, rasterized=True,
)
ax.set_xlim(-0.5, len(columns) - 0.5)
ax.set_ylim(len(VOC_ORDER) - 0.5, -0.5)
ax.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=9)
ax.set_yticks(range(len(VOC_ORDER)), VOC_ORDER, fontsize=9)
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_title("Confusion Matrix")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [LABELS[s] for s in labels], loc="upper center",
          bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
ax.tick_params(length=0)
save_plot("2_confusion_matrix.png")

# 3. Confusion heatmap for wrong-class predictions only
wrong = df[(df["status"] == "wrong_class") & df["det_cls"].notna()]
if not wrong.empty:
    matrix = pd.crosstab(wrong["cls"], wrong["det_cls"]).reindex(
        index=VOC_ORDER, columns=VOC_ORDER, fill_value=0
    )
    fig4, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(matrix, cmap="OrRd", annot=True, fmt="d", linewidths=0.25,
                linecolor="#eeeeee", cbar_kws={"label": "Wrong-class objects"}, ax=ax)
    ax.set_title(f"Mislabelled Predictions (n={len(wrong)})")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.tick_params(axis="x", rotation=60)
    ax.tick_params(axis="y", rotation=0)
    boundaries = np.cumsum([len(classes) for classes in CLASS_TYPES.values()])[:-1]
    for boundary in boundaries:
        ax.axhline(boundary, color="#606060", linewidth=1.8)
        ax.axvline(boundary, color="#606060", linewidth=1.8)
    save_plot("3_mislabelled_confusion_matrix.png")

# 4. Confidence vs IoU 

plot_df = df.copy()
plot_df["plot_iou"] = pd.to_numeric(plot_df["det_iou"], errors="coerce")
plot_df["plot_conf"] = pd.to_numeric(plot_df["conf"], errors="coerce")
plot_df["metric_source"] = "Matched detection"

if {"candidate_iou", "candidate_conf"}.issubset(plot_df.columns):
    missed = plot_df["status"].eq("missed")
    candidate_iou = pd.to_numeric(plot_df["candidate_iou"], errors="coerce")
    candidate_conf = pd.to_numeric(plot_df["candidate_conf"], errors="coerce")
    
    plot_df.loc[missed, "plot_iou"] = candidate_iou[missed].combine_first(
        plot_df.loc[missed, "plot_iou"]
    )
    plot_df.loc[missed, "plot_conf"] = candidate_conf[missed].combine_first(
        plot_df.loc[missed, "plot_conf"]
    )
    has_candidate = missed & candidate_iou.notna() & candidate_conf.notna()
    plot_df.loc[has_candidate, "metric_source"] = "Unmatched candidate"

missing_iou = plot_df["plot_iou"].isna()
missing_conf = plot_df["plot_conf"].isna()
plot_df.loc[missing_iou | missing_conf, "metric_source"] = "No available prediction metric"

rng = np.random.default_rng(42)
plot_df["display_iou"] = plot_df["plot_iou"].copy()
plot_df["display_conf"] = plot_df["plot_conf"].copy()
plot_df.loc[missing_iou, "display_iou"] = -0.075 + rng.uniform(-0.012, 0.012, missing_iou.sum())
plot_df.loc[missing_conf, "display_conf"] = -0.075 + rng.uniform(-0.012, 0.012, missing_conf.sum())
def make_thumbnail(row):
    image_path = IMAGE_DIR / f"{row['image']}.jpg"
    if not image_path.is_file():
        return None

    with Image.open(image_path) as source:
        original = source.convert("RGB")
        width, height = original.size
        scale = min(260 / width, 1.0)
        preview = original.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.LANCZOS,
        )

    sx, sy = preview.width / width, preview.height / height
    x1, y1, x2, y2 = [float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
    ImageDraw.Draw(preview).rectangle(
        (x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline="#29d16f", width=3
    )
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=68)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# 1. Interactive object-position scatter 
thumbnails = {}
for _, row in df.iterrows():
    thumbnail = make_thumbnail(row)
    if thumbnail:
        thumbnails[row["hover_id"]] = thumbnail

fig = go.Figure()
max_area = max(float(df["rel_area"].max()), 1e-6)
size_ref = 2 * max_area / (33 ** 2)

for status in STATUSES:
    part = df[df["status"] == status]
    if part.empty:
        continue

    custom = np.column_stack([
        part["image"].astype(str),
        part["cls"].astype(str),
        part["det_cls"].fillna("None").astype(str),
        part["size_bucket"].astype(str),
        part["rel_area"].mul(100).map(lambda x: f"{x:.2f}"),
        part["conf"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"),
        part["hover_id"].astype(str),
    ])
    fig.add_trace(go.Scattergl(
        x=part["center_x"], y=part["center_y"], mode="markers",
        name=f"{LABELS[status]} ({len(part)})",
        marker=dict(
            size=part["rel_area"].clip(lower=0),
            sizemode="area", sizeref=size_ref, sizemin=5,
            color=COLORS[status], opacity=0.35 if status == "correct" else 0.85,
        ),
        customdata=custom,
        hoverinfo="none",
        hovertemplate=None,
    ))

fig.update_layout(
    title="Object Position by Detection Outcome",
    template="plotly_white", width=900, height=760,
    hovermode="closest", legend=dict(orientation="h", y=1.07, x=0),
    margin=dict(l=80, r=40, t=105, b=75),
)
fig.update_xaxes(title="Horizontal image position (left to right)", range=[0, 1], dtick=0.2)
fig.update_yaxes(title="Vertical image position (top to bottom)", range=[1, 0],
                 dtick=0.2, scaleanchor="x", scaleratio=1)

post_script = r"""
(function() {
  const plot = document.getElementById('{plot_id}');
  const previews = __PREVIEWS__;
  const tip = document.createElement('div');
  tip.style.cssText = 'position:fixed;display:none;z-index:9999;pointer-events:none;max-width:290px;background:white;border:1px solid #d7dde5;border-radius:10px;padding:10px;box-shadow:0 5px 24px rgba(0,0,0,.22);font:13px Arial,sans-serif;color:#222;';
  const img = document.createElement('img');
  img.style.cssText = 'max-width:260px;max-height:210px;width:auto;height:auto;display:block;border-radius:5px;margin-bottom:7px;';
  tip.appendChild(img);
  document.body.appendChild(tip);
  function move(event) {
    if (!event) return;
    const w = tip.offsetWidth || 290, h = tip.offsetHeight || 230;
    tip.style.left = Math.max(8, Math.min(window.innerWidth-w-8, event.clientX+16))+'px';
    tip.style.top = Math.max(8, Math.min(window.innerHeight-h-8, event.clientY+16))+'px';
  }
  plot.on('plotly_hover', function(ev) {
    const point = ev.points && ev.points[0];
    if (!point || !point.customdata) return;
    const d = point.customdata;
    const src = previews[String(d[6])];
    if (!src) { tip.style.display = 'none'; return; }
    img.src = src;
    tip.style.display = 'block';
    move(ev.event);
  });
  plot.on('plotly_unhover', function() { tip.style.display = 'none'; });
  plot.addEventListener('mousemove', move);
  plot.addEventListener('mouseleave', function() { tip.style.display = 'none'; });
})();
""".replace("__PREVIEWS__", json.dumps(thumbnails))
fig.write_html(
    str(OUT / "5_scatter_size_confidence.html"),
    include_plotlyjs=True, full_html=True, post_script=post_script,
)
print("Saved:", OUT / "4_scatter_size_confidence.html")


# Interactive  position scatter.
fig8 = go.Figure()
for status in STATUSES:
    part = plot_df[plot_df["status"] == status]
    if part.empty:
        continue

    custom = np.column_stack([
        part["image"].astype(str),
        part["cls"].astype(str),
        part["det_cls"].fillna("None").astype(str),
        part["size_bucket"].astype(str),
        part["rel_area"].mul(100).map(lambda x: f"{x:.2f}"),
        part["plot_conf"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"),
        part["hover_id"].astype(str),
        part["plot_iou"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"),
        part["metric_source"].astype(str),
    ])
    fig8.add_trace(go.Scattergl(
        x=part["display_iou"],
        y=part["display_conf"],
        mode="markers",
        name=f"{LABELS[status]} ({len(part)})",
        marker=dict(size=7, color=COLORS[status],
                    opacity=0.40 if status == "correct" else 0.83),
        customdata=custom,
        hoverinfo="none",
        hovertemplate=None,
    ))

fig8.add_vline(x=0.5, line_dash="dash", line_color="gray")
fig8.add_vline(x=-0.025, line_dash="dot", line_color="#aaaaaa")
fig8.add_hline(y=-0.025, line_dash="dot", line_color="#aaaaaa")
fig8.update_layout(
    title="YOLO26n: IoU vs Confidence (including missed detections)",
    template="plotly_white", width=950, height=760, hovermode="closest",
    legend=dict(orientation="h", y=1.07, x=0),
    margin=dict(l=85, r=40, t=110, b=85),
    annotations=[dict(x=0.5, y=1.02, xref="x", yref="paper",
                      text="IoU threshold: 0.5", showarrow=False,
                      font=dict(size=11, color="gray"))],
)
axis_ticks = [-0.075, 0, 0.25, 0.5, 0.75, 1]
axis_labels = ["N/A", "0", "0.25", "0.5", "0.75", "1"]
fig8.update_xaxes(title="Intersection over Union (IoU)", range=[-0.11, 1.02],
                  tickvals=axis_ticks, ticktext=axis_labels)
fig8.update_yaxes(title="Prediction confidence", range=[-0.11, 1.02],
                  tickvals=axis_ticks, ticktext=axis_labels)

html_path = OUT / "5_iuo_scatter_plot.html"
fig8.write_html(str(html_path), include_plotlyjs=True, full_html=True,
                post_script=post_script)
print("Saved:", html_path)
print("Complete")
