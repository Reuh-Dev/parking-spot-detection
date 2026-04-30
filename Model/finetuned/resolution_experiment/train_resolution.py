# =============================================================================
# train_resolution.py
# -----------------------------------------------------------------------------
# PURPOSE : Resolution experiment — trains YOLOv8s at 320, 640, and 1280
#           with all other hyperparameters held constant.
#           Shows how input resolution affects detection accuracy on the
#           parking spot dataset (small spots, distant views, clarity levels).
#
# HOW TO RUN:
#   All resolutions:      python train_resolution.py
#   Single resolution:    python train_resolution.py --res 640
#
# OUTPUTS:
#   runs/res_320/weights/best.pt
#   runs/res_640/weights/best.pt
#   runs/res_1280/weights/best.pt
#
# To evaluate after training:
#   python Results/evaluate.py --weights <path_to_best.pt>
#
# NOTE: Training at 1280 requires significantly more GPU memory. If you hit an
#       out-of-memory error, reduce BATCH_SIZE from 16 to 8 for that resolution.
# =============================================================================

import os
import argparse
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIG
# =============================================================================

# Project root = Project1/ (4 levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
               os.path.dirname(os.path.abspath(__file__)))))

DATA_YAML   = os.path.join(PROJECT_ROOT, "Dataset", "data.yaml")
RUNS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

RESOLUTIONS = [320, 640, 1280]

# =============================================================================
# FIXED HYPERPARAMETERS — identical to main train.py except imgsz
# =============================================================================

EPOCHS     = 50
BATCH_SIZE = 16
WORKERS    = 4
DEVICE     = 0


# =============================================================================
# TRAINING
# =============================================================================

def train_resolution(imgsz):
    run_name = f"res_{imgsz}"

    print("\n" + "=" * 60)
    print(f"TRAINING — imgsz = {imgsz}")
    print("=" * 60)
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Image size : {imgsz}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Device     : GPU {DEVICE} ({torch.cuda.get_device_name(DEVICE)})\n")

    model = YOLO("yolov8s.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=imgsz,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=RUNS_DIR,
        name=run_name,
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

    best = os.path.join(RUNS_DIR, run_name, "weights", "best.pt")
    print(f"\n  Done. Best weights: {best}")
    return best


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Resolution experiment — YOLOv8s parking detection"
    )
    parser.add_argument(
        "--res", type=int, default=None, choices=RESOLUTIONS,
        help="Train a single resolution (320, 640, or 1280). Omit to train all.",
    )
    return parser.parse_args()


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

    args   = parse_args()
    targets = [args.res] if args.res else RESOLUTIONS

    trained = []
    for res in targets:
        best = train_resolution(res)
        trained.append((res, best))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for res, best in trained:
        print(f"  res_{res} weights : {best}")
    print()
    print("  Evaluate each resolution with:")
    print("    python Results/evaluate.py --weights <path_to_best.pt>")
    print("=" * 60)


if __name__ == "__main__":
    main()
