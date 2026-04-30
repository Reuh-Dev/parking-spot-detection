# =============================================================================
# evaluate.py
# -----------------------------------------------------------------------------
# PURPOSE : Evaluate and compare models on the parking spot dataset.
#           Outputs are organised into three sub-folders under Results/visuals/.
#
# MODES
# ─────
#   Default  — base YOLOv8s vs fine-tuned YOLOv8s
#       python Results/evaluate.py
#       → Results/visuals/base_vs_finetuned/
#           confusion_matrix_base.png
#           confusion_matrix_finetuned.png
#           metrics_comparison.png
#
#   Resolution experiment — 320 / 640 / 1280 comparison
#       python Results/evaluate.py --resolution
#       → Results/visuals/resolution_experiment/
#           metrics_table.png
#           splits_table.png
#
#   Architecture comparison — fine-tuned YOLOv8s vs fine-tuned RT-DETR-l
#       python Results/evaluate.py --rtdetr
#       → Results/visuals/yolov8s_vs_rtdetr/
#           metrics_comparison.png
#           splits_table.png
#
#   Failure cases — annotated test images where the model made mistakes
#       python Results/evaluate.py --failures
#       → Results/visuals/failure_cases/
#           failure_001_<img>.jpg  (missed detections, false positives, wrong class)
# =============================================================================

import os
import shutil
import tempfile
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO, RTDETR

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
    "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "Model", "finetuned",        "best_model.pt")
RTDETR_WEIGHTS  = os.path.join(PROJECT_ROOT, "Model", "finetuned_rtdetr", "best_model.pt")
BASE_WEIGHTS    = os.path.join(PROJECT_ROOT, "Model", "base",             "yolov8s.pt")
DATA_YAML       = os.path.join(PROJECT_ROOT, "Dataset", "data.yaml")
RESULTS_DIR     = os.path.dirname(os.path.abspath(__file__))

DIR_BASE_VS_FT   = os.path.join(RESULTS_DIR, "visuals", "base_vs_finetuned")
DIR_RESOLUTION   = os.path.join(RESULTS_DIR, "visuals", "resolution_experiment")
DIR_YOLO_VS_DETR = os.path.join(RESULTS_DIR, "visuals", "yolov8s_vs_rtdetr")
DIR_FAILURES     = os.path.join(RESULTS_DIR, "visuals", "failure_cases")

RES_EXP_DIR = os.path.join(PROJECT_ROOT, "Model", "finetuned", "resolution_experiment", "runs")
RES_MODELS  = [
    ("YOLOv8s  320×320",  os.path.join(RES_EXP_DIR, "res_320",  "weights", "best.pt"), 320),
    ("YOLOv8s  640×640",  DEFAULT_WEIGHTS,                                               640),
    ("YOLOv8s 1280×1280", os.path.join(RES_EXP_DIR, "res_1280", "weights", "best.pt"), 1280),
]

CLASS_NAMES   = ["empty", "occupied"]
METRIC_LABELS = [
    ("mAP@0.5",      "mAP50"),
    ("mAP@0.5:0.95", "mAP50_95"),
    ("Precision",    "precision"),
    ("Recall",       "recall"),
    ("F1-Score",     "f1"),
]

# =============================================================================
# HELPERS
# =============================================================================

def _run_val(weights_path, device, split="test", imgsz=640, conf=0.001, model_cls=YOLO):
    model  = model_cls(weights_path)
    tmpdir = tempfile.mkdtemp()
    try:
        metrics = model.val(
            data=DATA_YAML,
            split=split,
            imgsz=imgsz,
            device=device,
            verbose=False,
            project=tmpdir,
            conf=conf,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    p  = metrics.box.mp
    r  = metrics.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "mAP50":     metrics.box.map50,
        "mAP50_95":  metrics.box.map,
        "precision": p,
        "recall":    r,
        "f1":        f1,
    }, metrics


# =============================================================================
# CONFUSION MATRIX — fine-tuned model (2×2)
# =============================================================================

def plot_confusion_matrix(metrics, save_path, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    raw     = metrics.confusion_matrix.matrix
    cm_norm = raw / (raw.sum(axis=0, keepdims=True) + 1e-9)
    cm_norm = cm_norm[:2, :2]

    _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("True Label (Actual)", fontsize=12)
    ax.set_ylabel("Predicted Label",     fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# CONFUSION MATRIX — base model (COCO classes vs parking GT)
# =============================================================================

def _iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    b1 = [x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2]
    b2 = [x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2]
    ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter / (union + 1e-9)


def build_base_detection_matrix(device):
    model       = YOLO(BASE_WEIGHTS)
    test_images = os.path.join(PROJECT_ROOT, "Dataset", "test", "images")
    test_labels = os.path.join(PROJECT_ROOT, "Dataset", "test", "labels")
    matrix      = np.zeros((80, 2), dtype=np.int64)
    matrix      = np.zeros((80, 2), dtype=np.int64)

    image_files = sorted([
        f for f in os.listdir(test_images)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"  Running native COCO inference on {len(image_files)} test images...")

    for img_file in image_files:
        img_path   = os.path.join(test_images, img_file)
        label_path = os.path.join(test_labels, os.path.splitext(img_file)[0] + ".txt")

        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        if cls < 2:
                            gt_boxes.append({"cls": cls, "box": list(map(float, parts[1:5]))})
        if not gt_boxes:
            continue

        results = model.predict(img_path, verbose=False, device=device,
                                conf=0.1, iou=0.5, imgsz=640)
        preds = []
        if results[0].boxes is not None and len(results[0].boxes):
            for box in results[0].boxes:
                preds.append({"cls": int(box.cls[0]), "box": box.xywhn[0].tolist()})

        matched_preds = set()
        for gt in gt_boxes:
            best_iou, best_idx = 0.2, -1
            for i, pred in enumerate(preds):
                if i in matched_preds:
                    continue
                iou = _iou(gt["box"], pred["box"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx >= 0:
                coco_cls = preds[best_idx]["cls"]
                if coco_cls < 80:
                    matrix[coco_cls, gt["cls"]] += 1
                matched_preds.add(best_idx)

    return matrix


def plot_base_confusion_matrix(matrix, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    row_totals = matrix.sum(axis=1)
    nonzero    = np.where(row_totals > 0)[0]
    if len(nonzero) == 0:
        print("  [WARNING] Base model made no IoU-matched detections on test set.")
        return

    top_idx  = nonzero[np.argsort(-row_totals[nonzero])][:16]
    top_idx  = sorted(top_idx)
    cm       = matrix[top_idx, :]
    cm_norm  = cm / (cm.sum(axis=0, keepdims=True) + 1e-9)
    y_labels = [COCO_NAMES[i] for i in top_idx]

    fig_h = max(5, len(top_idx) * 0.5)
    _, ax = plt.subplots(figsize=(6, fig_h))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Reds",
        xticklabels=["empty (GT)", "occupied (GT)"],
        yticklabels=y_labels,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("True Parking Label (GT)", fontsize=12)
    ax.set_ylabel("Predicted by Base YOLOv8s (COCO)", fontsize=12)
    ax.set_title("Base YOLOv8s — Detections on Parking Images\n(no fine-tuning)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# METRICS TABLE (test-set comparison)
# =============================================================================

def plot_metrics_table(rows, save_path, title="Model Performance on Parking Spot Test Set"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    col_labels = ["Model", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1-Score"]
    keys       = ["mAP50", "mAP50_95", "precision", "recall", "f1"]
    cell_data  = [[name] + [f"{m[k]:.4f}" for k in keys] for name, m in rows]

    _, ax = plt.subplots(figsize=(13, 1.2 + len(rows) * 0.8))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)

    tbl = ax.table(
        cellText=cell_data, colLabels=col_labels,
        loc="center", cellLoc="center",
        colWidths=[0.34] + [0.11] * 5,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 2.2)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1f4e79")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        bg = "#dce6f1" if (i % 2 == 1) else "#ffffff"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
            if j > 0:
                tbl[i, j].set_text_props(fontfamily="monospace")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# SPLITS TABLE (train / val / test per model)
# =============================================================================

def plot_splits_table(rows, save_path, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    col_labels = ["Model", "Split", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1-Score"]
    keys       = ["mAP50", "mAP50_95", "precision", "recall", "f1"]
    cell_data  = [
        [r["model"], r["split"]] + [f"{r[k]:.4f}" for k in keys]
        for r in rows
    ]

    _, ax = plt.subplots(figsize=(16, 1.5 + len(rows) * 0.75))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)

    tbl = ax.table(
        cellText=cell_data, colLabels=col_labels,
        loc="center", cellLoc="center",
        colWidths=[0.22, 0.09] + [0.12] * 5,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 2.1)

    group_colors = ["#dce6f1", "#e8f4e8", "#fff3e0"]
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1f4e79")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        bg = group_colors[((i - 1) // 3) % len(group_colors)]
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
            if j > 1:
                tbl[i, j].set_text_props(fontfamily="monospace")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# MODE 1 — BASE vs FINE-TUNED YOLOv8s
# =============================================================================

def evaluate(weights_path):
    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("BASE vs FINE-TUNED YOLOv8s")
    print("=" * 60)

    print("\n  [1/3] Evaluating fine-tuned YOLOv8s...")
    m_ft, metrics_ft = _run_val(weights_path, device)

    print("\n  [2/3] Evaluating base YOLOv8s on parking classes...")
    m_base, _ = _run_val(BASE_WEIGHTS, device)

    print("\n  [3/3] Running base YOLOv8s natively (COCO classes)...")
    base_matrix = build_base_detection_matrix(device)

    plot_base_confusion_matrix(
        base_matrix,
        save_path=os.path.join(DIR_BASE_VS_FT, "confusion_matrix_base.png"),
    )
    plot_confusion_matrix(
        metrics_ft,
        save_path=os.path.join(DIR_BASE_VS_FT, "confusion_matrix_finetuned.png"),
        title="Confusion Matrix — Fine-tuned YOLOv8s",
    )
    plot_metrics_table(
        rows=[
            ("Base YOLOv8s (no fine-tuning)", m_base),
            ("Fine-tuned YOLOv8s",            m_ft),
        ],
        save_path=os.path.join(DIR_BASE_VS_FT, "metrics_comparison.png"),
        title="Base vs Fine-tuned YOLOv8s — Test Set Metrics",
    )

    print(f"\n  Output folder: {DIR_BASE_VS_FT}")


# =============================================================================
# MODE 2 — RESOLUTION EXPERIMENT
# =============================================================================

def resolution_experiment():
    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("RESOLUTION EXPERIMENT  (320 / 640 / 1280)")
    print("=" * 60)

    for label, path, _ in RES_MODELS:
        if not os.path.exists(path):
            print(f"  [ERROR] Weights not found: {path}")
            return

    print("\n  [1/2] Test-set metrics for all resolutions...")
    test_rows = []
    for i, (label, path, res) in enumerate(RES_MODELS, 1):
        print(f"    [{i}/3] {label}...")
        m, _ = _run_val(path, device, split="test", imgsz=res)
        test_rows.append((label, m))

    plot_metrics_table(
        rows=test_rows,
        save_path=os.path.join(DIR_RESOLUTION, "metrics_table.png"),
        title="Resolution Experiment — Test Set Metrics",
    )

    print("\n  [2/2] Train / Val / Test for all resolutions...")
    splits_rows = []
    for i, (label, path, res) in enumerate(RES_MODELS, 1):
        print(f"    [{i}/3] {label}...")
        for split in ("train", "val", "test"):
            print(f"      {split}...")
            m, _ = _run_val(path, device, split=split, imgsz=res)
            splits_rows.append({"model": label, "split": split.capitalize(), **m})

    plot_splits_table(
        rows=splits_rows,
        save_path=os.path.join(DIR_RESOLUTION, "splits_table.png"),
        title="Resolution Experiment — Train / Val / Test Metrics",
    )

    print(f"\n  Output folder: {DIR_RESOLUTION}")


# =============================================================================
# MODE 3 — YOLOv8s vs RT-DETR
# =============================================================================

def rtdetr_experiment():
    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("YOLOv8s vs RT-DETR-l")
    print("=" * 60)

    models = [
        ("YOLOv8s (fine-tuned)",       DEFAULT_WEIGHTS, YOLO),
        ("RT-DETR-l (fine-tuned)",     RTDETR_WEIGHTS,  RTDETR),
    ]
    for label, path, _ in models:
        if not os.path.exists(path):
            print(f"  [ERROR] Weights not found for {label}: {path}")
            return

    splits_rows = []
    test_rows   = []
    total       = len(models) * 3
    step        = 0

    for label, path, cls in models:
        for split in ("train", "val", "test"):
            step += 1
            print(f"  [{step}/{total}] {label} — {split}...")
            m, _ = _run_val(path, device, split=split, model_cls=cls)
            splits_rows.append({"model": label, "split": split.capitalize(), **m})
            if split == "test":
                test_rows.append((label, m))

    plot_metrics_table(
        rows=test_rows,
        save_path=os.path.join(DIR_YOLO_VS_DETR, "metrics_comparison.png"),
        title="YOLOv8s vs RT-DETR-l — Test Set Metrics",
    )
    plot_splits_table(
        rows=splits_rows,
        save_path=os.path.join(DIR_YOLO_VS_DETR, "splits_table.png"),
        title="YOLOv8s vs RT-DETR-l — Train / Val / Test Metrics",
    )

    print(f"\n  Output folder: {DIR_YOLO_VS_DETR}")


# =============================================================================
# MODE 4 — FAILURE CASE VISUALIZATION
# =============================================================================

def failure_cases(weights_path, max_images=30):
    import cv2

    os.makedirs(DIR_FAILURES, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"

    test_images = os.path.join(PROJECT_ROOT, "Dataset", "test", "images")
    test_labels = os.path.join(PROJECT_ROOT, "Dataset", "test", "labels")

    model = YOLO(weights_path)

    image_files = sorted([
        f for f in os.listdir(test_images)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print("=" * 60)
    print("FAILURE CASE VISUALIZATION")
    print("=" * 60)
    print(f"  Scanning {len(image_files)} test images for errors...\n")

    # colour scheme (BGR for OpenCV)
    COLORS = {
        "fn":    (255, 100,   0),   # blue-orange  — missed GT (false negative)
        "fp":    (0,   0,   220),   # red          — hallucinated box (false positive)
        "wrong": (0,  165,  255),   # orange       — right place, wrong class
    }

    saved = 0
    counts = {"fn": 0, "fp": 0, "wrong": 0}

    for img_file in image_files:
        if saved >= max_images:
            break

        img_path   = os.path.join(test_images, img_file)
        label_path = os.path.join(test_labels, os.path.splitext(img_file)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # load ground truth
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        gt_boxes.append({"cls": cls, "box": [x1, y1, x2, y2], "matched": False})

        if not gt_boxes:
            continue

        # run inference
        results = model.predict(img_path, conf=0.25, iou=0.5,
                                device=device, verbose=False, imgsz=640)
        preds = []
        if results[0].boxes is not None and len(results[0].boxes):
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                preds.append({
                    "cls": int(box.cls[0]),
                    "box": [x1, y1, x2, y2],
                    "matched": False,
                })

        # match predictions to GT (IoU >= 0.5)
        errors = []
        for pred in preds:
            px1, py1, px2, py2 = pred["box"]
            best_iou, best_gt = 0.0, None
            for gt in gt_boxes:
                if gt["matched"]:
                    continue
                gx1, gy1, gx2, gy2 = gt["box"]
                ix = max(0, min(px2, gx2) - max(px1, gx1))
                iy = max(0, min(py2, gy2) - max(py1, gy1))
                inter = ix * iy
                union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
                iou = inter / (union + 1e-9)
                if iou > best_iou:
                    best_iou, best_gt = iou, gt

            if best_iou >= 0.5 and best_gt is not None:
                best_gt["matched"] = True
                pred["matched"]    = True
                if pred["cls"] != best_gt["cls"]:
                    errors.append(("wrong", pred["box"], pred["cls"], best_gt["cls"]))
            else:
                errors.append(("fp", pred["box"], pred["cls"], None))

        for gt in gt_boxes:
            if not gt["matched"]:
                errors.append(("fn", gt["box"], None, gt["cls"]))

        if not errors:
            continue

        # draw on image
        canvas = img.copy()
        for err_type, box, pred_cls, gt_cls in errors:
            x1, y1, x2, y2 = box
            color = COLORS[err_type]

            if err_type == "fn":
                # dashed rectangle for missed GT
                for i in range(x1, x2, 10):
                    cv2.line(canvas, (i, y1), (min(i+5, x2), y1), color, 2)
                    cv2.line(canvas, (i, y2), (min(i+5, x2), y2), color, 2)
                for i in range(y1, y2, 10):
                    cv2.line(canvas, (x1, i), (x1, min(i+5, y2)), color, 2)
                    cv2.line(canvas, (x2, i), (x2, min(i+5, y2)), color, 2)
                label = f"MISSED: {CLASS_NAMES[gt_cls]}"
                counts["fn"] += 1
            elif err_type == "fp":
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                label = f"FALSE POS: {CLASS_NAMES[pred_cls]}"
                counts["fp"] += 1
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                label = f"WRONG: pred={CLASS_NAMES[pred_cls]} gt={CLASS_NAMES[gt_cls]}"
                counts["wrong"] += 1

            cv2.putText(canvas, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # legend
        legend = [
            ((255, 100, 0),  "Missed detection (FN)"),
            ((0,   0, 220),  "False positive (FP)"),
            ((0, 165, 255),  "Wrong class"),
        ]
        for i, (c, txt) in enumerate(legend):
            cv2.putText(canvas, txt, (8, 16 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)

        out_path = os.path.join(DIR_FAILURES, f"failure_{saved + 1:03d}_{img_file}")
        cv2.imwrite(out_path, canvas)
        saved += 1

    print(f"  Saved {saved} failure images to: {DIR_FAILURES}")
    print(f"  Missed detections (FN) : {counts['fn']}")
    print(f"  False positives  (FP)  : {counts['fp']}")
    print(f"  Wrong class            : {counts['wrong']}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Parking Spot Detection — Evaluation")
    parser.add_argument("--weights",    type=str, default=DEFAULT_WEIGHTS,
                        help="Path to fine-tuned YOLOv8s weights (default mode)")
    parser.add_argument("--data",       type=str, default=None,
                        help="Override data.yaml path")
    parser.add_argument("--resolution", action="store_true",
                        help="Run resolution experiment (320 / 640 / 1280)")
    parser.add_argument("--rtdetr",     action="store_true",
                        help="Compare fine-tuned RT-DETR-l vs fine-tuned YOLOv8s")
    parser.add_argument("--failures",   action="store_true",
                        help="Save annotated failure cases from the test set")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.data:
        global DATA_YAML
        DATA_YAML = args.data

    if args.resolution:
        resolution_experiment()
    elif args.rtdetr:
        rtdetr_experiment()
    elif args.failures:
        if not os.path.exists(args.weights):
            print(f"[ERROR] Weights not found: {args.weights}")
            return
        failure_cases(args.weights)
    else:
        if not os.path.exists(args.weights):
            print(f"[ERROR] Weights not found: {args.weights}")
            return
        evaluate(args.weights)


if __name__ == "__main__":
    main()
