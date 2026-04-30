# Parking Spot Detection — Computer Vision Project

Saint Joseph University · Computer Vision · Spring 2026

A real-time parking spot detection system that classifies individual parking spaces as **empty** or **occupied** using aerial/CCTV imagery. The project implements and compares three distinct approaches: a fine-tuned CNN (YOLOv8s), a fine-tuned Transformer (RT-DETR-l), and a classical computer vision baseline (edge density).

---

## Problem Statement

Manual monitoring of parking lots is inefficient and costly. This system automates the process by detecting and classifying every visible parking spot in a single forward pass, enabling real-time occupancy monitoring without infrastructure changes.

---

## Approaches

| Approach | Method | Description |
|---|---|---|
| YOLOv8s (fine-tuned) | Deep learning — CNN | Fine-tuned from COCO pretrained weights on the parking dataset |
| RT-DETR-l (fine-tuned) | Deep learning — Transformer | Fine-tuned transformer-based detector for architecture comparison |
| Classical CV baseline | Edge density | Manual ROI selection + Canny edge density thresholding, no training required |

---

## Dataset

- **Source:** [Roboflow — Parking Finder](https://universe.roboflow.com/aiml-the-lebron-project/parking-finder/dataset/1)
- **License:** CC BY 4.0
- **Classes:** 2 — `empty` (0), `occupied` (1)
- **Pre-processing (by Roboflow):** auto-orientation, resize to 640×640, rotation, noise, grayscale augmentation
- **Training augmentation (applied during training):** mosaic, HSV shifts, horizontal flip, rotation, scale, translation

| Split | Images | Empty instances | Occupied instances | Total instances |
|---|---|---|---|---|
| Train | 7,103 | 45,563 | 92,676 | 138,239 |
| Val | 558 | 3,650 | 8,300 | 11,950 |
| Test | 226 | 2,010 | 4,394 | 6,404 |
| **Total** | **7,887** | **51,223** | **105,370** | **156,593** |

> Occupied instances are approximately 2× more frequent than empty across all splits, reflecting real-world parking lot conditions.

---

## Project Structure

```
Project1/
├── Dataset/
│   ├── data.yaml                         — YOLO dataset config (classes, split paths)
│   ├── train/images/ + train/labels/     — 7,103 training images + YOLO labels
│   ├── valid/images/ + valid/labels/     — 558 validation images + YOLO labels
│   └── test/images/  + test/labels/      — 226 test images + YOLO labels
│
├── Model/
│   ├── base/
│   │   └── yolov8s.pt                    — COCO pretrained baseline (not fine-tuned)
│   ├── finetuned/
│   │   ├── train.py                      — Fine-tune YOLOv8s (50 epochs, 640px, batch 16)
│   │   ├── best_model.pt                 — Best fine-tuned YOLOv8s weights
│   │   └── resolution_experiment/
│   │       └── train_resolution.py       — Ablation: train at 320 / 640 / 1280
│   └── finetuned_rtdetr/
│       ├── train.py                      — Fine-tune RT-DETR-l (50 epochs, 640px, batch 8)
│       └── best_model.pt                 — Best fine-tuned RT-DETR-l weights
│
├── Demo/
│   ├── demo.py                           — Gradio web interface (3 tabs)
│   └── run_demo.bat                      — Windows launcher for the demo
│
├── Results/
│   ├── evaluate.py                       — All evaluation modes (see below)
│   └── visuals/
│       ├── base_vs_finetuned/            — Confusion matrices + metrics comparison
│       ├── resolution_experiment/        — 320 vs 640 vs 1280 metrics tables
│       ├── yolov8s_vs_rtdetr/            — Architecture comparison tables
│       └── failure_cases/               — Annotated test images showing model errors
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/Reuh-Dev/parking-spot-detection
cd Project1
pip install -r requirements.txt
```

> GPU strongly recommended for training. CPU is sufficient for inference and the demo.

### 2. Download the dataset

The dataset is already included in the `Dataset/` folder. If re-downloading, get it from [Roboflow](https://universe.roboflow.com/aiml-the-lebron-project/parking-finder/dataset/1) in YOLOv8 format and place it under `Dataset/`.

### 3. Download pretrained weights

Download the fine-tuned weights from the link below and place them at the paths shown:

| File | Path |
|---|---|
| Fine-tuned YOLOv8s | `Model/finetuned/best_model.pt` |
| Fine-tuned RT-DETR-l | `Model/finetuned_rtdetr/best_model.pt` |

**Download:** [Google Drive — Model Weights](<INSERT_DRIVE_LINK_HERE>)

---

## Running the Demo

The quickest way to see the system in action:

```bash
# Windows
Demo\run_demo.bat

# or directly
python Demo/demo.py
```

Then open **http://127.0.0.1:7860** in your browser.

The demo has three tabs:

| Tab | Description |
|---|---|
| Image Analysis | Upload a parking image — fine-tuned YOLOv8s detects and classifies every spot |
| Video Analysis | Upload a video — model samples frames at a configurable interval and shows annotated snapshots + parking maps |
| Manual ROI (Video) | Click to mark spot centers on a reference frame — classical CV edge density classifies each spot per snapshot |

---

## Training

> Requires a CUDA-capable GPU.

### Fine-tune YOLOv8s

```bash
python Model/finetuned/train.py
```

Trains for 50 epochs at 640×640, batch size 16. Best weights saved to `Model/finetuned/runs/yolov8s/weights/best.pt`.

### Fine-tune RT-DETR-l

```bash
python Model/finetuned_rtdetr/train.py
```

Trains for 50 epochs at 640×640, batch size 8. Best weights saved to `Model/finetuned_rtdetr/runs/rtdetr/weights/best.pt`.

### Resolution Ablation (320 / 640 / 1280)

```bash
# Train all three resolutions
python Model/finetuned/resolution_experiment/train_resolution.py

# Train a single resolution
python Model/finetuned/resolution_experiment/train_resolution.py --res 640
```

---

## Evaluation

All evaluation modes are handled by a single script with flags:

### Base YOLOv8s vs Fine-tuned YOLOv8s
```bash
python Results/evaluate.py
```
Outputs to `Results/visuals/base_vs_finetuned/`:
- `confusion_matrix_base.png` — what COCO classes the base model predicts on parking images
- `confusion_matrix_finetuned.png` — 2×2 empty/occupied confusion matrix
- `metrics_comparison.png` — mAP, precision, recall, F1 comparison table

### Resolution Experiment
```bash
python Results/evaluate.py --resolution
```
Outputs to `Results/visuals/resolution_experiment/`:
- `metrics_table.png` — test set metrics for all three resolutions
- `splits_table.png` — train/val/test metrics for all three resolutions

### YOLOv8s vs RT-DETR-l
```bash
python Results/evaluate.py --rtdetr
```
Outputs to `Results/visuals/yolov8s_vs_rtdetr/`:
- `metrics_comparison.png` — test set comparison table
- `splits_table.png` — train/val/test comparison for both models

### Failure Case Visualization
```bash
python Results/evaluate.py --failures
```
Outputs to `Results/visuals/failure_cases/`:
- Up to 30 annotated test images showing missed detections, false positives, and wrong-class predictions

---

## Design Choices & Experiments

| Experiment | What was tested | Conclusion |
|---|---|---|
| Base vs fine-tuned | COCO pretrained YOLOv8s vs fine-tuned on parking data | Fine-tuning is essential — base model has no concept of parking classes |
| Resolution ablation | 320×320 vs 640×640 vs 1280×1280 | 640px offers the best speed/accuracy trade-off for this dataset |
| Architecture comparison | CNN (YOLOv8s) vs Transformer (RT-DETR-l) | YOLOv8s is a better fit — parking spots are small, uniform, and locally classifiable |

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 and RT-DETR implementation
- [Roboflow](https://roboflow.com) — dataset hosting and preprocessing
- [Gradio](https://gradio.app) — demo interface framework
- Dataset: *Parking Finder* by AIML — The LeBron Project, licensed under CC BY 4.0
