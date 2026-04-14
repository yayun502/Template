import os
import csv
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from configs.config import (
    TRAIN_DIR, VAL_DIR, LABEL_MAP, IDX2LABEL,
    IMAGE_SIZE, MAX_LOCAL_VIEWS, MAX_GLOBAL_VIEWS, LOCAL_FOV_THRESHOLD,
    NUM_WORKERS, BATCH_SIZE,
    BACKBONE_TYPE, BACKBONE_NAME, LOCAL_PRETRAINED_PATH,
    FEAT_DIM, NUM_CLASSES, NUM_ATTRS,
    EPOCHS, LR, WEIGHT_DECAY, ATTR_LOSS_WEIGHT, DEVICE,
    SAVE_DIR, BEST_MODEL_NAME, LAST_MODEL_NAME,
    LOG_DIR, TRAIN_LOG_CSV
)
from data.dataset import DefectSampleDataset, get_sample_dirs
from models.defect_model import DefectClassifier
from utils.losses import compute_total_loss
from utils.metrics import compute_classification_metrics
from utils.seed import set_seed


def init_train_log(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_cls_loss",
                "train_attr_loss",
                "val_loss",
                "val_cls_loss",
                "val_attr_loss",
                "val_acc"
            ])


def append_train_log(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def train_one_epoch(model, loader, optimizer, device, attr_loss_weight):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_attr_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        local_imgs = batch["local_imgs"].to(device)
        global_imgs = batch["global_imgs"].to(device)
        local_mask = batch["local_mask"].to(device)
        global_mask = batch["global_mask"].to(device)
        labels = batch["label"].to(device)
        attr_labels = batch["attr_labels"].to(device)

        outputs = model(
            local_imgs=local_imgs,
            global_imgs=global_imgs,
            local_mask=local_mask,
            global_mask=global_mask
        )

        loss, cls_loss, attr_loss = compute_total_loss(
            outputs, labels, attr_labels, attr_loss_weight
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_attr_loss += attr_loss.item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "attr_loss": total_attr_loss / n
    }


@torch.no_grad()
def evaluate(model, loader, device, attr_loss_weight):
    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_attr_loss = 0.0

    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="Val", leave=False):
        local_imgs = batch["local_imgs"].to(device)
        global_imgs = batch["global_imgs"].to(device)
        local_mask = batch["local_mask"].to(device)
        global_mask = batch["global_mask"].to(device)
        labels = batch["label"].to(device)
        attr_labels = batch["attr_labels"].to(device)

        outputs = model(
            local_imgs=local_imgs,
            global_imgs=global_imgs,
            local_mask=local_mask,
            global_mask=global_mask
        )

        loss, cls_loss, attr_loss = compute_total_loss(
            outputs, labels, attr_labels, attr_loss_weight
        )

        preds = torch.argmax(outputs["logits"], dim=1)

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_attr_loss += attr_loss.item()

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    n = len(loader)
    metrics = compute_classification_metrics(
        y_true, y_pred,
        label_names=[IDX2LABEL[i] for i in range(len(IDX2LABEL))]
    )

    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "attr_loss": total_attr_loss / n,
        "metrics": metrics
    }


def main():
    set_seed(42)

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    init_train_log(TRAIN_LOG_CSV)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    train_dirs = get_sample_dirs(TRAIN_DIR)
    val_dirs = get_sample_dirs(VAL_DIR)

    train_dataset = DefectSampleDataset(
        sample_dirs=train_dirs,
        label_map=LABEL_MAP,
        image_size=IMAGE_SIZE,
        max_local=MAX_LOCAL_VIEWS,
        max_global=MAX_GLOBAL_VIEWS,
        local_fov_threshold=LOCAL_FOV_THRESHOLD,
        is_train=True
    )

    val_dataset = DefectSampleDataset(
        sample_dirs=val_dirs,
        label_map=LABEL_MAP,
        image_size=IMAGE_SIZE,
        max_local=MAX_LOCAL_VIEWS,
        max_global=MAX_GLOBAL_VIEWS,
        local_fov_threshold=LOCAL_FOV_THRESHOLD,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = DefectClassifier(
        backbone_type=BACKBONE_TYPE,
        backbone_name=BACKBONE_NAME,
        pretrained=(BACKBONE_TYPE == "timm"),
        pretrained_path=LOCAL_PRETRAINED_PATH if BACKBONE_TYPE == "resnet50_local" else None,
        feat_dim=FEAT_DIM,
        num_classes=NUM_CLASSES,
        num_attrs=NUM_ATTRS
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, ATTR_LOSS_WEIGHT
        )
        val_stats = evaluate(
            model, val_loader, device, ATTR_LOSS_WEIGHT
        )

        val_acc = val_stats["metrics"]["accuracy"]

        print(
            f"Train Loss: {train_stats['loss']:.4f} | "
            f"Cls: {train_stats['cls_loss']:.4f} | "
            f"Attr: {train_stats['attr_loss']:.4f}"
        )
        print(
            f"Val Loss:   {val_stats['loss']:.4f} | "
            f"Cls: {val_stats['cls_loss']:.4f} | "
            f"Attr: {val_stats['attr_loss']:.4f}"
        )
        print(f"Val Acc: {val_acc:.4f}")
        print("Classification Report:")
        print(val_stats["metrics"]["classification_report"])
        print("Confusion Matrix:")
        print(val_stats["metrics"]["confusion_matrix"])

        append_train_log(TRAIN_LOG_CSV, [
            epoch,
            train_stats["loss"],
            train_stats["cls_loss"],
            train_stats["attr_loss"],
            val_stats["loss"],
            val_stats["cls_loss"],
            val_stats["attr_loss"],
            val_acc
        ])

        last_path = os.path.join(SAVE_DIR, LAST_MODEL_NAME)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc
        }, last_path)

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(SAVE_DIR, BEST_MODEL_NAME)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc
            }, best_path)
            print(f"Saved best model to: {best_path}")


if __name__ == "__main__":
    main()
