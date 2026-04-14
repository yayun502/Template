import os
import csv

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from configs.config import (
    TEST_DIR, LABEL_MAP, IDX2LABEL,
    IMAGE_SIZE, MAX_LOCAL_VIEWS, MAX_GLOBAL_VIEWS, LOCAL_FOV_THRESHOLD,
    NUM_WORKERS, BATCH_SIZE,
    BACKBONE_TYPE, BACKBONE_NAME, LOCAL_PRETRAINED_PATH,
    FEAT_DIM, NUM_CLASSES, NUM_ATTRS, DEVICE,
    SAVE_DIR, BEST_MODEL_NAME,
    INFER_DIR, TEST_PRED_CSV, TEST_CM_PNG
)
from data.dataset import DefectSampleDataset, get_sample_dirs
from models.defect_model import DefectClassifier


def save_confusion_matrix(cm, class_names, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(INFER_DIR, exist_ok=True)

    test_dirs = get_sample_dirs(TEST_DIR)
    test_dataset = DefectSampleDataset(
        sample_dirs=test_dirs,
        label_map=LABEL_MAP,
        image_size=IMAGE_SIZE,
        max_local=MAX_LOCAL_VIEWS,
        max_global=MAX_GLOBAL_VIEWS,
        local_fov_threshold=LOCAL_FOV_THRESHOLD,
        is_train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = DefectClassifier(
        backbone_type=BACKBONE_TYPE,
        backbone_name=BACKBONE_NAME,
        pretrained=False,
        pretrained_path=LOCAL_PRETRAINED_PATH if BACKBONE_TYPE == "resnet50_local" else None,
        feat_dim=FEAT_DIM,
        num_classes=NUM_CLASSES,
        num_attrs=NUM_ATTRS
    ).to(device)

    ckpt_path = os.path.join(SAVE_DIR, BEST_MODEL_NAME)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_rows = []
    y_true = []
    y_pred = []

    for batch in tqdm(test_loader, desc="Infer"):
        local_imgs = batch["local_imgs"].to(device)
        global_imgs = batch["global_imgs"].to(device)
        local_mask = batch["local_mask"].to(device)
        global_mask = batch["global_mask"].to(device)
        labels = batch["label"].to(device)
        sample_dirs_batch = batch["sample_dir"]

        outputs = model(
            local_imgs=local_imgs,
            global_imgs=global_imgs,
            local_mask=local_mask,
            global_mask=global_mask
        )

        probs = torch.softmax(outputs["logits"], dim=1)
        confs, preds = torch.max(probs, dim=1)

        for i in range(len(sample_dirs_batch)):
            sample_name = os.path.basename(sample_dirs_batch[i])
            gt_idx = labels[i].item()
            pred_idx = preds[i].item()
            conf = confs[i].item()

            row = {
                "sample_name": sample_name,
                "gt_cls": IDX2LABEL[gt_idx],
                "pred_cls": IDX2LABEL[pred_idx],
                "conf": conf,
                "correct": int(gt_idx == pred_idx),
            }

            for cls_idx in range(NUM_CLASSES):
                row[f"prob_{IDX2LABEL[cls_idx]}"] = probs[i, cls_idx].item()

            all_rows.append(row)
            y_true.append(gt_idx)
            y_pred.append(pred_idx)

    # 寫 CSV
    fieldnames = [
        "sample_name", "gt_cls", "pred_cls", "conf", "correct"
    ] + [f"prob_{IDX2LABEL[i]}" for i in range(NUM_CLASSES)]

    with open(TEST_PRED_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # confusion matrix
    class_names = [IDX2LABEL[i] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    save_confusion_matrix(cm, class_names, TEST_CM_PNG)

    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)

    print("===== Inference Finished =====")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test samples: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Prediction CSV saved to: {TEST_PRED_CSV}")
    print(f"Confusion matrix figure saved to: {TEST_CM_PNG}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
