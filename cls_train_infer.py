"""
YOLOv8 Classification Training + Inference + Metrics

Assumes dataset already converted to:

output_dataset/
├── train/
│   ├── normal/
│   ├── defect/
├── val/
│   ├── normal/
│   ├── defect/

This script contains:
1. train()  -> train YOLOv8 classification model
2. infer_and_evaluate() -> run inference + compute TP/FP/TN/FN

Install:
pip install ultralytics scikit-learn
"""

import os
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# CONFIG
# =========================
DATASET_DIR = "output_dataset"
MODEL_NAME = "yolov8n-cls.pt"
TRAIN_EPOCHS = 100
IMG_SIZE = 224

# class mapping (IMPORTANT)
CLASS_NAMES = ["normal", "defect"]
CLASS_TO_ID = {"normal": 0, "defect": 1}

# =========================
# TRAIN
# =========================

def train():
    model = YOLO(MODEL_NAME)

    model.train(
        data=DATASET_DIR,
        epochs=TRAIN_EPOCHS,
        imgsz=IMG_SIZE,

        # augmentation
        fliplr=0.5,
        flipud=0.5,
        degrees=10,
        scale=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )


# =========================
# INFERENCE + METRICS
# =========================

def get_image_paths(split="val"):
    image_paths = []
    gt_labels = []

    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATASET_DIR, split, cls)
        if not os.path.exists(cls_dir):
            continue

        for fname in os.listdir(cls_dir):
            if fname.endswith(".jpg"):
                image_paths.append(os.path.join(cls_dir, fname))
                gt_labels.append(CLASS_TO_ID[cls])

    return image_paths, gt_labels


def infer_and_evaluate(model_path, split="val"):
    model = YOLO(model_path)

    image_paths, gt_labels = get_image_paths(split)

    preds = []

    for img_path in image_paths:
        results = model.predict(img_path, verbose=False)

        # top-1 prediction
        pred_class = int(results[0].probs.top1)
        preds.append(pred_class)

    # =========================
    # METRICS
    # =========================
    cm = confusion_matrix(gt_labels, preds)

    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(cm)

    print("\nTP, FP, TN, FN:")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    print("\nDerived Metrics:")
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(gt_labels, preds, target_names=CLASS_NAMES))


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--model", type=str, default="runs/classify/train/weights/best.pt")

    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "eval":
        infer_and_evaluate(args.model)
