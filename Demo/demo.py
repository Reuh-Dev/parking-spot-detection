# =============================================================================
# demo.py
# -----------------------------------------------------------------------------
# PURPOSE : Gradio web interface for parking spot detection.
#           Tab 1 — upload an image, fine-tuned YOLOv8s detects + classifies.
#           Tab 2 — upload a video, snapshots with detections + parking maps.
#           Tab 3 — manual ROI on video: click spots on a reference frame,
#                   base YOLOv8s (COCO) checks each region per snapshot.
#
# HOW TO RUN:
#   python Demo/demo.py
#   Then open http://127.0.0.1:7860 in your browser.
#
# REQUIREMENTS:
#   pip install gradio ultralytics opencv-python torch matplotlib
# =============================================================================

import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gradio as gr
from ultralytics import YOLO

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "Model", "finetuned", "best_model.pt")
BASE_WEIGHTS    = os.path.join(PROJECT_ROOT, "Model", "base", "yolov8s.pt")

CONF_THRESHOLD  = 0.15
IOU_THRESHOLD   = 0.5
EDGE_THRESHOLD  = 0.07   # fraction of edge pixels that marks a crop as "occupied"
CLASS_NAMES     = {0: "empty", 1: "occupied"}
COLORS_RGB      = {0: (34, 197, 94), 1: (239, 68, 68)}   # green / red


# =============================================================================
# MODELS — loaded once at startup
# =============================================================================

_model = YOLO(DEFAULT_WEIGHTS)   # fine-tuned, used by Tab 1 & 2


# =============================================================================
# SHARED DRAWING HELPERS
# =============================================================================

def annotate_image(image_rgb, results):
    img    = image_rgb.copy()
    counts = {0: 0, 1: 0}
    boxes  = results[0].boxes
    if boxes is not None and len(boxes):
        for box in boxes:
            cls             = int(box.cls[0])
            conf            = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color           = COLORS_RGB.get(cls, (255, 255, 0))
            name            = CLASS_NAMES.get(cls, str(cls))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{name} {conf:.2f}", (x1, max(y1 - 5, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            counts[cls] += 1
    return img, counts


def _fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf  = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    return buf[:, :, :3]


def _finish_map(fig, ax, empty_n, occupied_n, title):
    total = empty_n + occupied_n
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(title, color="white", fontsize=11, pad=8, fontweight="bold")
    fig.text(0.5, 0.01,
             f"Total: {total}    Empty: {empty_n}    Occupied: {occupied_n}",
             ha="center", color="#9ca3af", fontsize=9)
    for x, label, col in [(0.05, "■ Empty", "#22c55e"), (0.28, "■ Occupied", "#ef4444")]:
        fig.text(x, 0.95, label, color=col, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])


def make_parking_map(results, timestamp=None):
    boxes   = results[0].boxes
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#111827"); fig.patch.set_facecolor("#111827")
    empty_n = occupied_n = 0
    if boxes is not None and len(boxes):
        for i, box in enumerate(boxes):
            cls                 = int(box.cls[0])
            x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
            color = "#22c55e" if cls == 0 else "#ef4444"
            ax.add_patch(patches.FancyBboxPatch(
                (x1n, 1.0 - y2n), x2n - x1n, y2n - y1n,
                boxstyle="round,pad=0.005", linewidth=0.8,
                edgecolor="#374151", facecolor=color, alpha=0.85,
            ))
            ax.text((x1n + x2n) / 2, 1.0 - (y1n + y2n) / 2, str(i + 1),
                    ha="center", va="center", fontsize=5.5, color="white", fontweight="bold")
            if cls == 0: empty_n += 1
            else:        occupied_n += 1
    title = "Parking Map" + (f"  ·  t = {int(timestamp)}s" if timestamp is not None else "")
    _finish_map(fig, ax, empty_n, occupied_n, title)
    return _fig_to_rgb(fig)


def make_parking_map_rois(rois, statuses, img_h, img_w, timestamp=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#111827"); fig.patch.set_facecolor("#111827")
    empty_n = occupied_n = 0
    for i, ((x1, y1, x2, y2), occupied) in enumerate(zip(rois, statuses)):
        x1n, y1n = x1 / img_w, y1 / img_h
        x2n, y2n = x2 / img_w, y2 / img_h
        color = "#ef4444" if occupied else "#22c55e"
        ax.add_patch(patches.FancyBboxPatch(
            (x1n, 1.0 - y2n), x2n - x1n, y2n - y1n,
            boxstyle="round,pad=0.005", linewidth=0.8,
            edgecolor="#374151", facecolor=color, alpha=0.85,
        ))
        ax.text((x1n + x2n) / 2, 1.0 - (y1n + y2n) / 2, str(i + 1),
                ha="center", va="center", fontsize=5.5, color="white", fontweight="bold")
        if occupied: occupied_n += 1
        else:        empty_n += 1
    title = "Parking Map — Manual ROI" + (f"  ·  t = {int(timestamp)}s" if timestamp is not None else "")
    _finish_map(fig, ax, empty_n, occupied_n, title)
    return _fig_to_rgb(fig)


def _stats_html(e, o):
    t = e + o
    return (
        f"<div style='display:flex;gap:36px;font-size:1.25em;padding:14px 8px;font-family:sans-serif'>"
        f"<span>🅿️ <b>Total</b> &mdash; {t}</span>"
        f"<span style='color:#22c55e'>🟢 <b>Empty</b> &mdash; {e}</span>"
        f"<span style='color:#ef4444'>🔴 <b>Occupied</b> &mdash; {o}</span>"
        f"</div>"
    )


# =============================================================================
# TAB 1 — IMAGE ANALYSIS (fine-tuned model)
# =============================================================================

def run_image(image):
    if image is None:
        return None, None, "<p style='color:#6b7280'>Upload an image to begin.</p>"
    device  = 0 if torch.cuda.is_available() else "cpu"
    results = _model.predict(image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                             device=device, verbose=False, imgsz=640, save=False)
    annotated, counts = annotate_image(image, results)
    return annotated, make_parking_map(results), _stats_html(counts[0], counts[1])


# =============================================================================
# TAB 2 — VIDEO ANALYSIS (fine-tuned model)
# =============================================================================

def run_video(video_path, interval_sec, progress=gr.Progress()):
    if video_path is None:
        return [], []
    device     = 0 if torch.cuda.is_available() else "cpu"
    cap        = cv2.VideoCapture(video_path)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(fps * interval_sec))
    targets    = list(range(0, n_frames, frame_step))
    snapshots, maps = [], []
    for i, fidx in enumerate(targets):
        progress(i / len(targets), desc=f"Snapshot {i + 1} / {len(targets)}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok: break
        ts        = fidx / fps
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = _model.predict(frame_rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                                   device=device, verbose=False, imgsz=640, save=False)
        annotated, counts = annotate_image(frame_rgb, results)
        e, o = counts[0], counts[1]
        cv2.putText(annotated, f"t={int(ts)}s  |  Empty={e}  Occupied={o}",
                    (8, annotated.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        snapshots.append(annotated)
        maps.append(make_parking_map(results, timestamp=ts))
    cap.release()
    return snapshots, maps


# =============================================================================
# TAB 3 — MANUAL ROI + VIDEO (classical CV — edge density, no ML)
# =============================================================================

def _is_occupied(crop):
    """Edge-density check: parked cars have far more edges than empty asphalt."""
    gray    = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges   = cv2.Canny(gray, 50, 150)
    density = edges.sum() / (255.0 * edges.size)
    return density > EDGE_THRESHOLD

def _draw_spots(frame, spots, box_size):
    """Overlay numbered orange boxes on the reference frame."""
    if frame is None:
        return None
    disp = frame.copy()
    half = box_size // 2
    for i, (cx, cy) in enumerate(spots):
        x1 = max(0, cx - half);          y1 = max(0, cy - half)
        x2 = min(disp.shape[1], cx + half); y2 = min(disp.shape[0], cy + half)
        cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.circle(disp, (cx, cy), 5, (255, 165, 0), -1)
        cv2.putText(disp, str(i + 1), (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1, cv2.LINE_AA)
    return disp


def t3_load_frame(video_path):
    """Extract first frame from video to use as the ROI reference."""
    if video_path is None:
        return None, None, []
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, None, []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb, rgb, []   # display, orig (State), spots (State)


def t3_add_spot(frame_orig, spots, box_size, evt: gr.SelectData):
    """Add a clicked spot and redraw the reference frame."""
    if frame_orig is None:
        return None, spots
    x, y   = int(evt.index[0]), int(evt.index[1])
    spots  = spots + [(x, y)]
    return _draw_spots(frame_orig, spots, box_size), spots


def t3_clear_spots(frame_orig):
    """Remove all marked spots."""
    return frame_orig, []


def t3_update_boxes(frame_orig, spots, box_size):
    """Redraw boxes when slider changes."""
    return _draw_spots(frame_orig, spots, box_size)


def run_video_roi(video_path, frame_orig, spots, box_size, interval_sec,
                  progress=gr.Progress()):
    """Process video with manually-defined ROIs using base YOLOv8s."""
    if video_path is None or frame_orig is None or not spots:
        return [], []

    ref_h, ref_w = frame_orig.shape[:2]
    half = box_size // 2
    rois = []
    for (cx, cy) in spots:
        x1, y1 = max(0, cx - half),   max(0, cy - half)
        x2, y2 = min(ref_w, cx + half), min(ref_h, cy + half)
        if x2 > x1 and y2 > y1:
            rois.append((x1, y1, x2, y2))

    if not rois:
        return [], []

    cap        = cv2.VideoCapture(video_path)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(fps * interval_sec))
    targets    = list(range(0, n_frames, frame_step))

    snapshots, maps = [], []

    for i, fidx in enumerate(targets):
        progress(i / len(targets), desc=f"Snapshot {i + 1} / {len(targets)}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok: break

        ts        = fidx / fps
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw    = frame_rgb.shape[:2]
        result_img = frame_rgb.copy()
        statuses   = []

        for j, (x1, y1, x2, y2) in enumerate(rois):
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(fw, x2), min(fh, y2)
            crop     = frame_rgb[y1c:y2c, x1c:x2c]

            occupied = False
            if crop.shape[0] >= 10 and crop.shape[1] >= 10:
                occupied = _is_occupied(crop)

            statuses.append(occupied)
            color = COLORS_RGB[1] if occupied else COLORS_RGB[0]
            label = "occupied" if occupied else "empty"
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_img, f"#{j + 1} {label}",
                        (x1, max(y1 - 5, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        e, o = statuses.count(False), statuses.count(True)
        cv2.putText(result_img, f"t={int(ts)}s  |  Empty={e}  Occupied={o}",
                    (8, result_img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        snapshots.append(result_img)
        maps.append(make_parking_map_rois(rois, statuses, fh, fw, timestamp=ts))

    cap.release()
    return snapshots, maps


# =============================================================================
# GRADIO UI
# =============================================================================

css = """
h1 { text-align:center; margin-bottom:2px; }
.subtitle { text-align:center; color:#6b7280; margin-bottom:16px; font-size:0.95em; }
"""

_default_stats = "<p style='color:#6b7280;padding:8px'>Upload an image and click Analyze.</p>"

with gr.Blocks(title="Parking Spot Detection") as app:

    gr.HTML("<h1>🅿️ Parking Spot Detection</h1>")
    gr.HTML("<p class='subtitle'>Fine-tuned YOLOv8s &nbsp;·&nbsp; "
            "Empty / Occupied Classification &nbsp;·&nbsp; Manual ROI Baseline</p>")

    with gr.Tabs():

        # ── Tab 1: Image ──────────────────────────────────────────────────────
        with gr.Tab("📷  Image Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    t1_input = gr.Image(label="Upload Parking Image",
                                        type="numpy", height=360)
                    t1_btn   = gr.Button("Analyze", variant="primary", size="lg")
                with gr.Column(scale=1):
                    t1_output = gr.Image(label="Detection Output", height=360, value=None)
            t1_stats = gr.HTML(value=_default_stats)
            t1_map   = gr.Image(label="Parking Map", height=280, value=None)
            t1_btn.click(run_image, inputs=t1_input,
                         outputs=[t1_output, t1_map, t1_stats])

        # ── Tab 2: Video ──────────────────────────────────────────────────────
        with gr.Tab("🎥  Video Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2_video    = gr.Video(label="Upload Parking Video")
                    t2_interval = gr.Slider(minimum=5, maximum=60, value=10, step=5,
                                            label="Take a snapshot every (seconds)")
                    t2_btn = gr.Button("Analyze Video", variant="primary", size="lg")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Annotated Snapshots")
                    t2_snapshots = gr.Gallery(label="Snapshots", columns=3,
                                              height=380, object_fit="contain")
                with gr.Column():
                    gr.Markdown("### Parking Maps")
                    t2_maps = gr.Gallery(label="Parking Maps", columns=3,
                                         height=380, object_fit="contain")
            t2_btn.click(run_video, inputs=[t2_video, t2_interval],
                         outputs=[t2_snapshots, t2_maps])

        # ── Tab 3: Manual ROI + Video ─────────────────────────────────────────
        with gr.Tab("✏️  Manual ROI (Video)"):
            gr.Markdown(
                "**How to use:**\n"
                "1. Upload the same parking video and click **Load Reference Frame**\n"
                "2. **Click on each parking space** in the image to mark its center "
                "(orange box appears)\n"
                "3. Adjust **Box Size** until the boxes fit the spaces, then click "
                "**Analyze Video**\n\n"
                "> **vs Tab 2 — Fine-tuned model:** Tab 2 uses a trained neural network "
                "that automatically finds and classifies every spot. This approach uses "
                "**classical computer vision (edge density)** — no ML, no training data — "
                "but requires manually marking each space once per parking lot."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    t3_video    = gr.Video(label="Upload Parking Video")
                    t3_load_btn = gr.Button("Load Reference Frame", variant="secondary")
                    t3_box_size = gr.Slider(minimum=30, maximum=300, value=100, step=10,
                                            label="Box Size (pixels)")
                    t3_interval = gr.Slider(minimum=5, maximum=60, value=10, step=5,
                                            label="Take a snapshot every (seconds)")
                    with gr.Row():
                        t3_clear_btn = gr.Button("Clear Spots", variant="secondary")
                        t3_run_btn   = gr.Button("Analyze Video",
                                                  variant="primary", size="lg")
                with gr.Column(scale=1):
                    t3_frame_display = gr.Image(
                        label="Reference Frame — click to mark parking spaces",
                        type="numpy", height=420, interactive=True, value=None,
                    )

            t3_frame_orig = gr.State(None)   # clean original frame
            t3_spots      = gr.State([])     # list of (x, y) centers

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Annotated Snapshots — Manual ROI")
                    t3_snapshots = gr.Gallery(label="Snapshots", columns=3,
                                              height=380, object_fit="contain")
                with gr.Column():
                    gr.Markdown("### Parking Maps — Manual ROI")
                    t3_maps = gr.Gallery(label="Parking Maps", columns=3,
                                         height=380, object_fit="contain")

            # Events
            t3_load_btn.click(
                t3_load_frame,
                inputs=t3_video,
                outputs=[t3_frame_display, t3_frame_orig, t3_spots],
            )
            t3_frame_display.select(
                t3_add_spot,
                inputs=[t3_frame_orig, t3_spots, t3_box_size],
                outputs=[t3_frame_display, t3_spots],
            )
            t3_box_size.change(
                t3_update_boxes,
                inputs=[t3_frame_orig, t3_spots, t3_box_size],
                outputs=t3_frame_display,
            )
            t3_clear_btn.click(
                t3_clear_spots,
                inputs=t3_frame_orig,
                outputs=[t3_frame_display, t3_spots],
            )
            t3_run_btn.click(
                run_video_roi,
                inputs=[t3_video, t3_frame_orig, t3_spots, t3_box_size, t3_interval],
                outputs=[t3_snapshots, t3_maps],
            )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft(), css=css, inbrowser=True)
