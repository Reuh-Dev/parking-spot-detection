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
│   ├── README.dataset.txt                — Dataset source info
│   └── README.roboflow.txt               — Roboflow export info
│   (images not included — download from Roboflow, see Setup below)
│
├── Model/
│   ├── base/
│   │   └── yolov8s.pt                    — COCO pretrained baseline (not fine-tuned)
│   ├── finetuned/
│   │   ├── train.py                      — Fine-tune YOLOv8s (50 epochs, 640px, batch 16)
│   │   ├── best_model.pt                 — Best fine-tuned YOLOv8s weights
│   │   └── resolution_experiment/
│   │       ├── train_resolution.py       — Ablation: train at 320 and 1280
│   │       ├── res_320/
│   │       │   └── best_model.pt         — Best weights at 320×320
│   │       └── res_1280/
│   │           └── best_model.pt         — Best weights at 1280×1280
│   └── finetuned_rtdetr/
│       ├── train.py                      — Fine-tune RT-DETR-l (50 epochs, 640px, batch 8)
│       └── best_model.pt                 — Best fine-tuned RT-DETR-l weights
│
├── Demo/
│   ├── demo.py                           — Gradio web interface (3 tabs)
│   └── run_demo.bat                      — Windows launcher for the demo
│
├── results/
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
cd parking-spot-detection
pip install -r requirements.txt
```

> GPU strongly recommended for training. CPU is sufficient for inference and the demo.

### 2. Download the dataset

The dataset images are not included in the repository due to size. Download the dataset in **YOLOv8 format** from Roboflow:

[Roboflow — Parking Finder Dataset](https://universe.roboflow.com/aiml-the-lebron-project/parking-finder/dataset/1)

Once downloaded, place the contents so the structure matches:

```
Dataset/
├── train/images/   and   train/labels/
├── valid/images/   and   valid/labels/
└── test/images/    and   test/labels/
```

The `data.yaml` file is already included in the repo and requires no changes.

### 3. Weights

All trained weights are included in the repository — no separate download needed:

| Model | Path |
|---|---|
| Base YOLOv8s (COCO pretrained) | `Model/base/yolov8s.pt` |
| Fine-tuned YOLOv8s | `Model/finetuned/best_model.pt` |
| Fine-tuned YOLOv8s 320×320 | `Model/finetuned/resolution_experiment/res_320/best_model.pt` |
| Fine-tuned YOLOv8s 1280×1280 | `Model/finetuned/resolution_experiment/res_1280/best_model.pt` |
| Fine-tuned RT-DETR-l | `Model/finetuned_rtdetr/best_model.pt` |

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

> Requires a CUDA-capable GPU. Complete the dataset setup first.

### Fine-tune YOLOv8s

```bash
python Model/finetuned/train.py
```

Trains for 50 epochs at 640×640, batch size 16. Best weights saved to `Model/finetuned/best_model.pt`.

### Fine-tune RT-DETR-l

```bash
python Model/finetuned_rtdetr/train.py
```

Trains for 50 epochs at 640×640, batch size 8. Best weights saved to `Model/finetuned_rtdetr/best_model.pt`.

### Resolution Ablation (320 and 1280)

```bash
# Train both resolutions
python Model/finetuned/resolution_experiment/train_resolution.py

# Train a single resolution
python Model/finetuned/resolution_experiment/train_resolution.py --res 320
python Model/finetuned/resolution_experiment/train_resolution.py --res 1280
```

Best weights saved to `Model/finetuned/resolution_experiment/res_320/best_model.pt` and `res_1280/best_model.pt`. The 640×640 baseline uses the main fine-tuned model directly.

---

## Evaluation

All evaluation modes are handled by a single script with flags:

### Base YOLOv8s vs Fine-tuned YOLOv8s
```bash
python results/evaluate.py
```
Outputs to `results/visuals/base_vs_finetuned/`:
- `confusion_matrix_base.png` — what COCO classes the base model predicts on parking images
- `confusion_matrix_finetuned.png` — 2×2 empty/occupied confusion matrix
- `metrics_comparison.png` — mAP, precision, recall, F1 comparison table

### Resolution Experiment
```bash
python results/evaluate.py --resolution
```
Outputs to `results/visuals/resolution_experiment/`:
- `metrics_table.png` — test set metrics for all three resolutions
- `splits_table.png` — train/val/test metrics for all three resolutions

### YOLOv8s vs RT-DETR-l
```bash
python results/evaluate.py --rtdetr
```
Outputs to `results/visuals/yolov8s_vs_rtdetr/`:
- `metrics_comparison.png` — test set comparison table
- `splits_table.png` — train/val/test comparison for both models

### Failure Case Visualization
```bash
python results/evaluate.py --failures
```
Outputs to `results/visuals/failure_cases/`:
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
