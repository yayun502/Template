import os
import json
import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# ============================================================
# Load Ground Truth
# ============================================================
def build_gt_dict(dataset_root, split_json_path, split_name, class_map):
    """
    Load ground truth from dataset directory.

    Structure:
        dataset/
            defect1/
                img1.jpg
                img1.json
            ...
            normal/
                imgX.jpg

    Expected json:
        {
        "objects": [
            {
            "label": "defectA",
            "bbox": [x1, y1, x2, y2]
            }
        ]
        }

    Returns:
        GT = {
            class_id: {
                image_id: [bbox1, bbox2, ...]   # bbox = [x1,y1,x2,y2]
            }
        }
    """

    GT = defaultdict(lambda: defaultdict(list))

    # load split mapping
    with open(split_json_path, "r") as f:
        split_data = json.load(f)

    target_files = split_data[split_name]

    # walk through dataset
    for root, dirs, files in os.walk(dataset_root):
        for file in files:

            if not file.endswith(".json"):
                continue

            image_id = file.replace(".json", "")

            if image_id not in target_files:
                continue

            json_path = os.path.join(root, file)

            with open(json_path, "r") as f:
                data = json.load(f)

            objects = data.get("objects", [])

            for obj in objects:

                label = obj["label"]

                if label not in class_map:
                    continue

                class_id = class_map[label]
                bbox = obj["bbox"]  # [x1,y1,x2,y2]

                GT[class_id][image_id].append(bbox)

    return GT


# ============================================================
# Load Predictions
# ============================================================
def build_pred_dict(pred_label_dir, split_json_path, split_name, conf_thresh=0.0):
    """
    Load prediction results.

    Each .txt file contains:
        class_id x1 y1 x2 y2 score
    
    Return:
        Pred = { 
            class_id: 
                { image_id: [box, box, ...] } # box = [x1,y1,x2,y2,confidence]
        } 
    """
    Pred = defaultdict(lambda: defaultdict(list))

    with open(split_json_path, "r") as f:
        split_data = json.load(f)

    target_files = set(split_data[split_name])

    for file in os.listdir(pred_label_dir):

        if not file.endswith(".txt"):
            continue

        image_id = file.replace(".txt", "")

        if image_id not in target_files:
            continue

        file_path = os.path.join(pred_label_dir, file)

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            class_id = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:5])
            score = float(parts[5])

            if score >= conf_thresh:
                Pred[class_id][image_id].append(
                    [x1, y1, x2, y2, score]
                )

    return Pred



# ============================================================
# IoU
# ============================================================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# NMS
# ============================================================

def nms(boxes, iou_thresh=0.5):
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)

        boxes = [
            box for box in boxes
            if compute_iou(best[:4], box[:4]) < iou_thresh
        ]

    return keep


# ============================================================
# AP (COCO 101-point interpolation)
# ============================================================

def compute_ap(recall, precision):
    recall_points = np.linspace(0, 1, 101)
    precision_interp = []

    for r in recall_points:
        p = precision[recall >= r]
        precision_interp.append(np.max(p) if p.size > 0 else 0)

    return np.mean(precision_interp)


# ============================================================
# Main evaluation
# ============================================================

def evaluate(GT, Pred, iou_thresholds=None, nms_iou=0.5):

    if iou_thresholds is None:
        # 保留 0.5~0.95 架構，但預設只使用 0.5
        iou_thresholds = [0.5]

    classes = sorted(GT.keys())
    num_classes = len(classes)

    # ========================================================
    # Apply NMS
    # ========================================================

    for cls in Pred:
        for img_id in Pred[cls]:
            Pred[cls][img_id] = nms(Pred[cls][img_id], nms_iou)

    # ========================================================
    # Image-level Escape / Overkill
    # ========================================================

    escape_img = 0
    overkill_img = 0
    total_defect_images = 0
    total_normal_images = 0

    all_images = set()
    for cls in GT:
        all_images.update(GT[cls].keys())
    for cls in Pred:
        all_images.update(Pred[cls].keys())

    for img in all_images:

        gt_exist = any(img in GT[c] and len(GT[c][img]) > 0 for c in GT)
        pred_exist = any(img in Pred[c] and len(Pred[c][img]) > 0 for c in Pred)

        if gt_exist:
            total_defect_images += 1
            if not pred_exist:
                escape_img += 1
        else:
            total_normal_images += 1
            if pred_exist:
                overkill_img += 1

    # ========================================================
    # BBox-level metrics + Confusion Matrix
    # ========================================================

    # Confusion matrix: (num_classes + 1) x (num_classes + 1)
    # 最後一列/行 = background
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1))

    per_class_results = {}
    aps_all_iou = []

    for iou_thresh in iou_thresholds:

        aps = []

        for cls_idx, cls in enumerate(classes):

            gt_cls = GT.get(cls, {})
            pred_cls = Pred.get(cls, {})

            total_gt = sum(len(v) for v in gt_cls.values())
            if total_gt == 0:
                continue

            all_preds = []
            for img_id in pred_cls:
                for box in pred_cls[img_id]:
                    all_preds.append((img_id, box))

            all_preds.sort(key=lambda x: x[1][4], reverse=True)

            tp = np.zeros(len(all_preds))
            fp = np.zeros(len(all_preds))

            matched = {
                img: np.zeros(len(gt_cls.get(img, [])))
                for img in gt_cls
            }

            for i, (img_id, pred_box) in enumerate(all_preds):

                gt_boxes = gt_cls.get(img_id, [])
                best_iou = 0
                best_gt_idx = -1

                for j, gt_box in enumerate(gt_boxes):
                    iou = compute_iou(pred_box[:4], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    if matched[img_id][best_gt_idx] == 0:
                        tp[i] = 1
                        matched[img_id][best_gt_idx] = 1
                        conf_matrix[cls_idx][cls_idx] += 1  # TP
                    else:
                        fp[i] = 1
                        conf_matrix[num_classes][cls_idx] += 1  # duplicate FP
                else:
                    fp[i] = 1
                    conf_matrix[num_classes][cls_idx] += 1  # background -> predicted

            # Count FN
            for img_id in gt_cls:
                for k in range(len(gt_cls[img_id])):
                    if matched[img_id][k] == 0:
                        conf_matrix[cls_idx][num_classes] += 1  # GT -> background

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)

            recall = tp_cum / total_gt
            precision = tp_cum / (tp_cum + fp_cum + 1e-6)

            ap = compute_ap(recall, precision)
            aps.append(ap)

            precision_final = precision[-1] if len(precision) > 0 else 0
            recall_final = recall[-1] if len(recall) > 0 else 0

            per_class_results[cls] = {
                "AP@0.5": ap,
                "Precision": precision_final,
                "Recall": recall_final,
                "Escape(1-Recall)": 1 - recall_final,
                "Overkill(1-Precision)": 1 - precision_final
            }

        if len(aps) > 0:
            aps_all_iou.append(np.mean(aps))

    mAP50 = aps_all_iou[0] if len(aps_all_iou) > 0 else 0

    # ========================================================
    # Final Results
    # ========================================================

    results = {
        "mAP@0.5": mAP50,
        "per_class": per_class_results,
        "confusion_matrix": conf_matrix,
        "image_level": {
            "Escape": escape_img,
            "Overkill": overkill_img,
            "Escape_rate": escape_img / total_defect_images if total_defect_images > 0 else 0,
            "Overkill_rate": overkill_img / total_normal_images if total_normal_images > 0 else 0
        }
    }

    return results


# ============================================================
# Main Entry
# ============================================================
if __name__ == "__main__":

    gt_root = "dataset"
    pred_root = "predictions"


    class_map = {
        "defectA": 0,
        "defectB": 1
    }

    GT = build_gt_dict(
        dataset_root="datasets/dataset",
        split_json_path="split.json",
        split_name="val",
        class_map=class_map
    )

    Pred = build_pred_dict(
        pred_label_dir="runs/valid_output/labels",
        split_json_path="split.json",
        split_name="val",
        conf_thresh=0.25
    )

    results = evaluate(GT, Pred)
    print(results)

    # print("\n===== Evaluation Result =====")
    # for k, v in results.items():
    #     print(f"{k}: {v:.4f}")