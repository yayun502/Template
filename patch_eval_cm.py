import os
import numpy as np

def compute_iou(box1, box2):
    """Compute IoU of two boxes [x1,y1,x2,y2]"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def load_gt(file):
    """Load GT txt: class x1 y1 x2 y2"""
    boxes = []
    if not os.path.exists(file):
        return boxes
    with open(file) as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:5])
            boxes.append({"class": class_id, "bbox": [x1, y1, x2, y2]})
    return boxes

def load_pred(file):
    """Load prediction txt: class x1 y1 x2 y2 confidence"""
    boxes = []
    if not os.path.exists(file):
        return boxes
    with open(file) as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:5])
            conf = float(parts[5])
            boxes.append({"class": class_id, "bbox": [x1, y1, x2, y2], "conf": conf})
    return boxes

def evaluate_image(gt_boxes, pred_boxes, confusion_matrix, num_classes, iou_thr=0.5):
    """
    Evaluate one image
    - TP/FP/FN
    - update confusion matrix (last row/col = background)
    """
    TP = 0
    FP = 0

    gt_used = [False] * len(gt_boxes)
    pred_boxes = sorted(pred_boxes, key=lambda x: x["conf"], reverse=True)

    for pred in pred_boxes:
        pred_class = pred["class"]
        pred_box = pred["bbox"]

        best_iou = 0
        best_gt_idx = -1

        # match only with same class GT
        for i, gt in enumerate(gt_boxes):
            if gt_used[i]:
                continue
            if gt["class"] != pred_class:
                continue
            iou = compute_iou(pred_box, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thr and best_gt_idx != -1:
            # matched GT
            TP += 1
            gt_used[best_gt_idx] = True
            confusion_matrix[pred_class, pred_class] += 1
        else:
            # FP (background)
            FP += 1
            confusion_matrix[num_classes, pred_class] += 1  # bg row

    # remaining GT = FN
    FN = 0
    for i, used in enumerate(gt_used):
        if not used:
            FN += 1
            gt_class = gt_boxes[i]["class"]
            confusion_matrix[gt_class, num_classes] += 1  # bg col

    return TP, FP, FN, confusion_matrix


def evaluate_dataset_per_class(gt_dir, pred_dir, num_classes, iou_thr=0.5):
    """
    Evaluate all images and provide per-class metrics
    """
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    total_TP = 0
    total_FP = 0
    total_FN = 0

    files = os.listdir(gt_dir)
    files = [f for f in files if f.endswith(".txt")]

    for file in files:
        gt_boxes = load_gt(os.path.join(gt_dir, file))
        pred_boxes = load_pred(os.path.join(pred_dir, file))
        TP, FP, FN, confusion_matrix = evaluate_image(
            gt_boxes, pred_boxes, confusion_matrix, num_classes, iou_thr
        )
        total_TP += TP
        total_FP += FP
        total_FN += FN

    # macro metrics
    precision = total_TP / (total_TP + total_FP + 1e-9)
    recall = total_TP / (total_TP + total_FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    escape_rate = total_FN / (total_TP + total_FN + 1e-9)
    overskill_rate = total_FP / (total_TP + total_FP + 1e-9)

    # per-class metrics
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    for c in range(num_classes):
        TPc = confusion_matrix[c, c]
        FPc = confusion_matrix[:, c].sum() - TPc
        FNc = confusion_matrix[c, :].sum() - TPc

        prec_c = TPc / (TPc + FPc + 1e-9)
        rec_c = TPc / (TPc + FNc + 1e-9)
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-9)

        per_class_precision.append(prec_c)
        per_class_recall.append(rec_c)
        per_class_f1.append(f1_c)

    results = {
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "escape_rate": escape_rate,
        "overskill_rate": overskill_rate,
        "confusion_matrix": confusion_matrix,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_F1": per_class_f1
    }

    return results

if __name__ == "__main__":
    
    num_classes = 5
    result = evaluate_dataset_per_class(
        gt_dir="labels",
        pred_dir="predictions",
        num_classes=num_classes,
        iou_thr=0.5
    )


    print("Per-class precision:", result["per_class_precision"])
    print("Per-class recall:", result["per_class_recall"])
    print("Per-class F1:", result["per_class_F1"])

    print("Precision:", result["precision"])
    print("Recall:", result["recall"])
    print("F1:", result["F1"])
    print("Escape rate (FN/GT):", result["escape_rate"])
    print("Overskill rate (FP/Pred):", result["overskill_rate"])
    print("Confusion matrix:\n", result["confusion_matrix"])