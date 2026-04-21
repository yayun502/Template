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
    BACKBONE_TYPE, BACKBONE_NAME, LOCAL_PRETRAINED_PATH, DINO_REPO_DIR,
    FEAT_DIM, NUM_CLASSES, DEVICE,
    SAVE_DIR, BEST_MODEL_NAME,
    INFER_DIR, TEST_PRED_CSV, TEST_CM_PNG,
    ATTN_CSV, ATTN_FIG_DIR,
    INFER_PRED_HEAD
)
from data.dataset import DefectSampleDataset, get_sample_dirs
from models.defect_model import DefectClassifier
from utils.hierarchical import hierarchical_probs_from_logits
from utils.sample_ordering import load_and_prepare_sample_items


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


def load_sample_meta(sample_dir, local_fov_threshold, max_local, max_global):
    """
    與 dataset.py 完全共用相同的排序 / fallback / truncate 邏輯。
    """
    local_items, global_items = load_and_prepare_sample_items(
        sample_dir=sample_dir,
        local_fov_threshold=local_fov_threshold,
        max_local=max_local,
        max_global=max_global
    )
    return local_items, global_items


def save_attention_bar_chart(sample_name, rows, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    local_rows = [r for r in rows if r["branch"] == "local"]
    global_rows = [r for r in rows if r["branch"] == "global"]

    n_local = len(local_rows)
    n_global = len(global_rows)

    if n_local == 0 and n_global == 0:
        return

    fig_h = 4 + max(n_local, n_global) * 0.4
    plt.figure(figsize=(12, fig_h))

    if n_local > 0:
        plt.subplot(2, 1, 1)
        labels = [f'{r["image_name"]}\n(FOV={r["fov"]})' for r in local_rows]
        values = [r["attention_weight"] for r in local_rows]
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), labels, rotation=45, ha="right")
        plt.ylabel("Attention")
        plt.title(f"{sample_name} - Local Branch Attention")

    if n_global > 0:
        plt.subplot(2, 1, 2)
        labels = [f'{r["image_name"]}\n(FOV={r["fov"]})' for r in global_rows]
        values = [r["attention_weight"] for r in global_rows]
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), labels, rotation=45, ha="right")
        plt.ylabel("Attention")
        plt.title(f"{sample_name} - Global Branch Attention")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{sample_name}_attention.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(INFER_DIR, exist_ok=True)
    os.makedirs(ATTN_FIG_DIR, exist_ok=True)

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
        pretrained_path=LOCAL_PRETRAINED_PATH,
        feat_dim=FEAT_DIM,
        num_classes=NUM_CLASSES,
        dino_repo_dir=DINO_REPO_DIR if BACKBONE_TYPE == "dinov2_local" else None
    ).to(device)

    ckpt_path = os.path.join(SAVE_DIR, BEST_MODEL_NAME)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_pred_rows = []
    all_attn_rows = []

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

        main_probs = torch.softmax(outputs["logits"], dim=1)
        hier_probs = hierarchical_probs_from_logits(outputs["hier_logits"])

        gate_probs = torch.sigmoid(outputs["hier_logits"])
        p_np = gate_probs[:, 0]
        p_single = gate_probs[:, 1]
        p_bp = gate_probs[:, 2]

        if INFER_PRED_HEAD == "main":
            final_probs = main_probs
        elif INFER_PRED_HEAD == "hier":
            final_probs = hier_probs
        else:
            raise ValueError(f"Unsupported INFER_PRED_HEAD: {INFER_PRED_HEAD}")

        confs, preds = torch.max(final_probs, dim=1)

        local_attn = outputs["local_attn"].cpu()
        global_attn = outputs["global_attn"].cpu()
        local_mask_cpu = local_mask.cpu()
        global_mask_cpu = global_mask.cpu()

        for i in range(len(sample_dirs_batch)):
            sample_dir = sample_dirs_batch[i]
            sample_name = os.path.basename(sample_dir)

            gt_idx = labels[i].item()
            pred_idx = preds[i].item()
            conf = confs[i].item()

            pred_row = {
                "sample_name": sample_name,
                "gt_cls": IDX2LABEL[gt_idx],
                "pred_cls": IDX2LABEL[pred_idx],
                "conf": conf,
                "correct": int(gt_idx == pred_idx),
                "pred_head": INFER_PRED_HEAD,
                "p_np": float(p_np[i].item()),
                "p_single": float(p_single[i].item()),
                "p_bp": float(p_bp[i].item())
            }

            for cls_idx in range(NUM_CLASSES):
                pred_row[f"main_prob_{IDX2LABEL[cls_idx]}"] = float(main_probs[i, cls_idx].item())
                pred_row[f"hier_prob_{IDX2LABEL[cls_idx]}"] = float(hier_probs[i, cls_idx].item())
                pred_row[f"final_prob_{IDX2LABEL[cls_idx]}"] = float(final_probs[i, cls_idx].item())

            all_pred_rows.append(pred_row)
            y_true.append(gt_idx)
            y_pred.append(pred_idx)

            local_items, global_items = load_sample_meta(
                sample_dir=sample_dir,
                local_fov_threshold=LOCAL_FOV_THRESHOLD,
                max_local=MAX_LOCAL_VIEWS,
                max_global=MAX_GLOBAL_VIEWS
            )

            sample_attn_rows = []

            local_valid_count = int(local_mask_cpu[i].sum().item())
            for j in range(local_valid_count):
                meta_item = local_items[j]
                row = {
                    "sample_name": sample_name,
                    "gt_cls": IDX2LABEL[gt_idx],
                    "pred_cls": IDX2LABEL[pred_idx],
                    "branch": "local",
                    "image_name": meta_item["image_name"],
                    "fov": meta_item["fov"],
                    "attention_weight": float(local_attn[i, j].item())
                }
                all_attn_rows.append(row)
                sample_attn_rows.append(row)

            global_valid_count = int(global_mask_cpu[i].sum().item())
            for j in range(global_valid_count):
                meta_item = global_items[j]
                row = {
                    "sample_name": sample_name,
                    "gt_cls": IDX2LABEL[gt_idx],
                    "pred_cls": IDX2LABEL[pred_idx],
                    "branch": "global",
                    "image_name": meta_item["image_name"],
                    "fov": meta_item["fov"],
                    "attention_weight": float(global_attn[i, j].item())
                }
                all_attn_rows.append(row)
                sample_attn_rows.append(row)

            save_attention_bar_chart(sample_name, sample_attn_rows, ATTN_FIG_DIR)

    pred_fieldnames = [
        "sample_name", "gt_cls", "pred_cls", "conf", "correct", "pred_head",
        "p_np", "p_single", "p_bp"
    ]
    pred_fieldnames += [f"main_prob_{IDX2LABEL[i]}" for i in range(NUM_CLASSES)]
    pred_fieldnames += [f"hier_prob_{IDX2LABEL[i]}" for i in range(NUM_CLASSES)]
    pred_fieldnames += [f"final_prob_{IDX2LABEL[i]}" for i in range(NUM_CLASSES)]

    with open(TEST_PRED_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fieldnames)
        writer.writeheader()
        writer.writerows(all_pred_rows)

    attn_fieldnames = [
        "sample_name", "gt_cls", "pred_cls",
        "branch", "image_name", "fov", "attention_weight"
    ]

    with open(ATTN_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=attn_fieldnames)
        writer.writeheader()
        writer.writerows(all_attn_rows)

    class_names = [IDX2LABEL[i] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    save_confusion_matrix(cm, class_names, TEST_CM_PNG)

    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)

    print("===== Inference Finished =====")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Prediction head: {INFER_PRED_HEAD}")
    print(f"Test samples: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Prediction CSV saved to: {TEST_PRED_CSV}")
    print(f"Attention CSV saved to: {ATTN_CSV}")
    print(f"Attention figures saved to: {ATTN_FIG_DIR}")
    print(f"Confusion matrix figure saved to: {TEST_CM_PNG}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
