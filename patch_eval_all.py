import os
import numpy as np

# --------------------------
# 基本工具
# --------------------------
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
    return inter_area / union if union > 0 else 0.0

def load_gt(file):
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

# --------------------------
# evaluate single image
# --------------------------
def evaluate_image(gt_boxes, pred_boxes, confusion_matrix, num_classes, iou_thr=0.5):
    TP = 0
    FP = 0
    gt_used = [False]*len(gt_boxes)
    pred_boxes = sorted(pred_boxes, key=lambda x: x["conf"], reverse=True)

    for pred in pred_boxes:
        pred_class = pred["class"]
        pred_box = pred["bbox"]

        best_iou = 0
        best_gt_idx = -1

        # per-class matching (VOC/COCO style)
        for i, gt in enumerate(gt_boxes):
            if gt_used[i] or gt["class"] != pred_class:
                continue
            iou = compute_iou(pred_box, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thr and best_gt_idx != -1:
            TP += 1
            gt_used[best_gt_idx] = True
            confusion_matrix[pred_class, pred_class] += 1
        else:
            FP += 1
            confusion_matrix[num_classes, pred_class] += 1  # background row

    # remaining GT = FN
    FN = 0
    for i, used in enumerate(gt_used):
        if not used:
            FN += 1
            gt_class = gt_boxes[i]["class"]
            confusion_matrix[gt_class, num_classes] += 1  # background col

    return TP, FP, FN, confusion_matrix

# --------------------------
# evaluate dataset (confusion matrix + per-class metrics)
# --------------------------
def evaluate_dataset(gt_dir, pred_dir, num_classes, iou_thr=0.5):
    confusion_matrix = np.zeros((num_classes+1, num_classes+1), dtype=int)
    total_TP = 0
    total_FP = 0
    total_FN = 0

    all_gts = []
    all_preds = []

    files = [f for f in os.listdir(gt_dir) if f.endswith(".txt")]
    for file in files:
        gt_boxes = load_gt(os.path.join(gt_dir, file))
        pred_boxes = load_pred(os.path.join(pred_dir, file))

        # save for AP/mAP calculation
        gts = {"boxes":[b["bbox"] for b in gt_boxes], "class":[b["class"] for b in gt_boxes]}
        preds = {"boxes":[b["bbox"] for b in pred_boxes],
                 "class":[b["class"] for b in pred_boxes],
                 "conf":[b["conf"] for b in pred_boxes]}
        all_gts.append(gts)
        all_preds.append(preds)

        TP, FP, FN, confusion_matrix = evaluate_image(
            gt_boxes, pred_boxes, confusion_matrix, num_classes, iou_thr
        )
        total_TP += TP
        total_FP += FP
        total_FN += FN

    # macro metrics
    precision = total_TP / (total_TP + total_FP + 1e-9)
    recall = total_TP / (total_TP + total_FN + 1e-9)
    f1 = 2*precision*recall/(precision+recall+1e-9)
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
        prec_c = TPc/(TPc+FPc+1e-9)
        rec_c = TPc/(TPc+FNc+1e-9)
        f1_c = 2*prec_c*rec_c/(prec_c+rec_c+1e-9)
        per_class_precision.append(prec_c)
        per_class_recall.append(rec_c)
        per_class_f1.append(f1_c)

    results = {
        "TP": total_TP, "FP": total_FP, "FN": total_FN,
        "precision": precision, "recall": recall, "F1": f1,
        "escape_rate": escape_rate, "overskill_rate": overskill_rate,
        "confusion_matrix": confusion_matrix,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_F1": per_class_f1,
        "all_preds": all_preds,
        "all_gts": all_gts
    }

    return results

# --------------------------
# AP / mAP calculation YOLOv8-style
# --------------------------
def compute_ap_mAP_yolo_style(all_preds, all_gts, num_classes, iou_thresholds=[0.5]):
    """
    Compute AP and mAP following YOLOv8 / COCO style (full PR curve, trapezoidal integration)
    """
    results = {}
    
    for iou_thr in iou_thresholds:
        ap_per_class = []

        for c in range(num_classes):
            preds_c = []
            n_gt = 0

            # 收集所有圖片該 class 的 GT 與 predictions
            for preds, gts in zip(all_preds, all_gts):
                gt_boxes_c = [b for b, cls in zip(gts['boxes'], gts['class']) if cls==c]
                n_gt += len(gt_boxes_c)
                pred_boxes_c = [(b, conf) for b, cls, conf in zip(preds['boxes'], preds['class'], preds['conf']) if cls==c]
                preds_c.extend([{'box':b, 'conf':conf, 'matched':False, 'gt_boxes':gt_boxes_c.copy()} for b, conf in pred_boxes_c])

            if n_gt == 0:
                ap_per_class.append(0.0)
                continue

            # 按 confidence 排序（高→低）
            preds_c.sort(key=lambda x:x['conf'], reverse=True)

            tp = np.zeros(len(preds_c))
            fp = np.zeros(len(preds_c))

            # 每個 prediction 只與本張圖片 GT match
            for i, pred in enumerate(preds_c):
                best_iou = 0
                best_gt_idx = -1
                for j, gt_box in enumerate(pred['gt_boxes']):
                    if gt_box is None:
                        continue
                    iou = compute_iou(pred['box'], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                if best_iou >= iou_thr and best_gt_idx != -1:
                    tp[i] = 1
                    pred['gt_boxes'][best_gt_idx] = None  # 標記 GT 已被 match
                else:
                    fp[i] = 1

            # 累積 TP / FP
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / n_gt
            prec = tp_cum / (tp_cum + fp_cum + 1e-9)

            # 梯形積分計算 AP（YOLO/COCO-style）
            ap = np.trapz(prec, rec)
            ap_per_class.append(ap)

        mAP = np.mean(ap_per_class)
        results[iou_thr] = (ap_per_class, mAP)

    return results

# --------------------------
# Example usage
# --------------------------
if __name__=="__main__":
    num_classes = 5
    gt_dir = "labels"
    pred_dir = "predictions"

    # Evaluate dataset (confusion matrix + per-class metrics)
    results = evaluate_dataset(gt_dir, pred_dir, num_classes=num_classes, iou_thr=0.5)
    print("Macro Precision:", results["precision"])
    print("Macro Recall:", results["recall"])
    print("Macro F1:", results["F1"])
    print("Per-class Precision:", results["per_class_precision"])
    print("Per-class Recall:", results["per_class_recall"])
    print("Per-class F1:", results["per_class_F1"])
    print("Confusion matrix:\n", results["confusion_matrix"])
    print("Escape rate:", results["escape_rate"])
    print("Overskill rate:", results["overskill_rate"])

    # AP/mAP YOLOv8-style
    iou_thresholds = [0.5]  # 可換成 [0.5, 0.55, ..., 0.95] 計算 mAP@0.5:0.95
    ap_results = compute_ap_mAP_yolo_style(results["all_preds"], results["all_gts"], num_classes=num_classes, iou_thresholds=iou_thresholds)
    for iou_thr, (ap_per_class, mAP) in ap_results.items():
        print(f"mAP@{iou_thr}: {mAP:.4f}")
        print("AP per class:", ["%.4f"%a for a in ap_per_class])