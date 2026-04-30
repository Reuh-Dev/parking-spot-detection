# =============================================================================
# train.py
# -----------------------------------------------------------------------------
# PURPOSE : Fine-tune YOLOv8s on the parking spot dataset (empty / occupied).
#           Starts from COCO pretrained weights and trains for 50 epochs with
#           data augmentation. Best weights are saved under runs/yolov8s/
#
# HOW TO RUN:
#   python train.py          (requires a CUDA-capable GPU)
#
# To evaluate the trained model run:
#   python Results/evaluate.py
# =============================================================================

import os
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_YAML = os.path.join(PROJECT_ROOT, "Dataset", "data.yaml")
RUNS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

EPOCHS     = 50
IMG_SIZE   = 640
BATCH_SIZE = 16
WORKERS    = 4
DEVICE     = 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [ERROR] CUDA not available. Training requires a GPU.")
        return

    print(f"  GPU detected : {torch.cuda.get_device_name(0)}")
    print(f"  data.yaml    : {DATA_YAML}")
    print(f"  Runs folder  : {RUNS_DIR}")

    if not os.path.exists(DATA_YAML):
        print(f"\n  [ERROR] data.yaml not found at: {DATA_YAML}")
        return

    print("\n" + "=" * 60)
    print("TRAINING YOLOv8s")
    print("=" * 60)
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Image size : {IMG_SIZE}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Device     : GPU {DEVICE} ({torch.cuda.get_device_name(DEVICE)})\n")

    model = YOLO("yolov8s.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=RUNS_DIR,
        name="yolov8s",
        exist_ok=True,
        pretrained=True,
        patience=20,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        plots=False,
    )

    best = os.path.join(RUNS_DIR, "yolov8s", "weights", "best.pt")
    print("\n" + "=" * 60)
    print(f"  Training complete.")
    print(f"  Best weights : {best}")
    print(f"  Run Results/evaluate.py to assess performance on the test set.")
    print("=" * 60)


if __name__ == "__main__":
    main()
