import os
import csv
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

from configs.config import (
    TRAIN_DIR, VAL_DIR, LABEL_MAP, IDX2LABEL,
    IMAGE_SIZE, MAX_LOCAL_VIEWS, MAX_GLOBAL_VIEWS, LOCAL_FOV_THRESHOLD,
    NUM_WORKERS, BATCH_SIZE,
    BACKBONE_TYPE, BACKBONE_NAME, LOCAL_PRETRAINED_PATH, DINO_REPO_DIR,
    FEAT_DIM, NUM_CLASSES,
    EPOCHS, LR, WEIGHT_DECAY, DEVICE,
    SAVE_DIR, BEST_MODEL_NAME, LAST_MODEL_NAME,
    LOG_DIR, TRAIN_LOG_CSV,
    USE_CLASS_WEIGHTS, CLASS_WEIGHTS,
    CLS_LOSS_TYPE, FOCAL_GAMMA,
    USE_HIERARCHICAL_HEAD, HIER_LOSS_WEIGHT, GATE_LOSS_WEIGHT,
    SCHEDULER_TYPE,
    STEP_SIZE, STEP_GAMMA,
    PLATEAU_MODE, PLATEAU_FACTOR, PLATEAU_PATIENCE,
    COSINE_T_MAX, COSINE_ETA_MIN
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
                "lr",
                "train_loss",
                "train_cls_loss",
                "train_hier_loss",
                "train_gate_loss",
                "val_loss",
                "val_cls_loss",
                "val_hier_loss",
                "val_gate_loss",
                "val_acc"
            ])


def append_train_log(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def build_scheduler(optimizer):
    if SCHEDULER_TYPE == "none":
        scheduler = None
    elif SCHEDULER_TYPE == "step":
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA)
    elif SCHEDULER_TYPE == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=COSINE_T_MAX, eta_min=COSINE_ETA_MIN)
    elif SCHEDULER_TYPE == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=PLATEAU_MODE,
            factor=PLATEAU_FACTOR,
            patience=PLATEAU_PATIENCE
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {SCHEDULER_TYPE}")
    return scheduler


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    class_weights=None,
    cls_loss_type="ce",
    focal_gamma=2.0,
    use_hierarchical=False,
    hier_loss_weight=0.5,
    gate_loss_weight=0.3
):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_hier_loss = 0.0
    total_gate_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        local_imgs = batch["local_imgs"].to(device)
        global_imgs = batch["global_imgs"].to(device)
        local_mask = batch["local_mask"].to(device)
        global_mask = batch["global_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            local_imgs=local_imgs,
            global_imgs=global_imgs,
            local_mask=local_mask,
            global_mask=global_mask
        )

        loss, cls_loss, hier_loss, gate_loss = compute_total_loss(
            outputs=outputs,
            class_labels=labels,
            class_weights=class_weights,
            cls_loss_type=cls_loss_type,
            focal_gamma=focal_gamma,
            use_hierarchical=use_hierarchical,
            hier_loss_weight=hier_loss_weight,
            gate_loss_weight=gate_loss_weight
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_hier_loss += hier_loss.item()
        total_gate_loss += gate_loss.item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "hier_loss": total_hier_loss / n,
        "gate_loss": total_gate_loss / n
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    class_weights=None,
    cls_loss_type="ce",
    focal_gamma=2.0,
    use_hierarchical=False,
    hier_loss_weight=0.5,
    gate_loss_weight=0.3
):
    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_hier_loss = 0.0
    total_gate_loss = 0.0

    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="Val", leave=False):
        local_imgs = batch["local_imgs"].to(device)
        global_imgs = batch["global_imgs"].to(device)
        local_mask = batch["local_mask"].to(device)
        global_mask = batch["global_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            local_imgs=local_imgs,
            global_imgs=global_imgs,
            local_mask=local_mask,
            global_mask=global_mask
        )

        loss, cls_loss, hier_loss, gate_loss = compute_total_loss(
            outputs=outputs,
            class_labels=labels,
            class_weights=class_weights,
            cls_loss_type=cls_loss_type,
            focal_gamma=focal_gamma,
            use_hierarchical=use_hierarchical,
            hier_loss_weight=hier_loss_weight,
            gate_loss_weight=gate_loss_weight
        )

        preds = torch.argmax(outputs["logits"], dim=1)

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_hier_loss += hier_loss.item()
        total_gate_loss += gate_loss.item()

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
        "hier_loss": total_hier_loss / n,
        "gate_loss": total_gate_loss / n,
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
        pretrained=False,
        pretrained_path=LOCAL_PRETRAINED_PATH,
        feat_dim=FEAT_DIM,
        num_classes=NUM_CLASSES,
        dino_repo_dir=DINO_REPO_DIR if BACKBONE_TYPE == "dinov2_local" else None
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer)

    print(f"Backbone type: {BACKBONE_TYPE}")
    print(f"Backbone name: {BACKBONE_NAME}")
    print(f"Scheduler type: {SCHEDULER_TYPE}")
    print(f"Classification loss type: {CLS_LOSS_TYPE}")
    if CLS_LOSS_TYPE == "focal":
        print(f"Focal gamma: {FOCAL_GAMMA}")

    print(f"Use hierarchical head: {USE_HIERARCHICAL_HEAD}")
    if USE_HIERARCHICAL_HEAD:
        print(f"Hier loss weight: {HIER_LOSS_WEIGHT}")
        print(f"Gate loss weight: {GATE_LOSS_WEIGHT}")

    if USE_CLASS_WEIGHTS:
        class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
        print(f"Using class weights: {CLASS_WEIGHTS}")
    else:
        class_weights = None
        print("Not using class weights.")

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            class_weights=class_weights,
            cls_loss_type=CLS_LOSS_TYPE,
            focal_gamma=FOCAL_GAMMA,
            use_hierarchical=USE_HIERARCHICAL_HEAD,
            hier_loss_weight=HIER_LOSS_WEIGHT,
            gate_loss_weight=GATE_LOSS_WEIGHT
        )

        val_stats = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            class_weights=class_weights,
            cls_loss_type=CLS_LOSS_TYPE,
            focal_gamma=FOCAL_GAMMA,
            use_hierarchical=USE_HIERARCHICAL_HEAD,
            hier_loss_weight=HIER_LOSS_WEIGHT,
            gate_loss_weight=GATE_LOSS_WEIGHT
        )

        val_acc = val_stats["metrics"]["accuracy"]

        if scheduler is not None:
            if SCHEDULER_TYPE == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Current LR: {current_lr:.8f}")
        print(
            f"Train Loss: {train_stats['loss']:.4f} | "
            f"Cls: {train_stats['cls_loss']:.4f} | "
            f"Hier: {train_stats['hier_loss']:.4f} | "
            f"Gate: {train_stats['gate_loss']:.4f}"
        )
        print(
            f"Val Loss:   {val_stats['loss']:.4f} | "
            f"Cls: {val_stats['cls_loss']:.4f} | "
            f"Hier: {val_stats['hier_loss']:.4f} | "
            f"Gate: {val_stats['gate_loss']:.4f}"
        )
        print(f"Val Acc: {val_acc:.4f}")
        print("Classification Report:")
        print(val_stats["metrics"]["classification_report"])
        print("Confusion Matrix:")
        print(val_stats["metrics"]["confusion_matrix"])

        append_train_log(TRAIN_LOG_CSV, [
            epoch,
            current_lr,
            train_stats["loss"],
            train_stats["cls_loss"],
            train_stats["hier_loss"],
            train_stats["gate_loss"],
            val_stats["loss"],
            val_stats["cls_loss"],
            val_stats["hier_loss"],
            val_stats["gate_loss"],
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
