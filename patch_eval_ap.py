import numpy as np


def compute_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter

    if union == 0:
        return 0.0

    return inter / union


def evaluate_class(gt_boxes, pred_boxes, class_id, iou_thr=0.5):
    """
    YOLO-style per-class matching
    """

    gt = [g for g in gt_boxes if g["class"] == class_id]
    pred = [p for p in pred_boxes if p["class"] == class_id]

    n_gt = len(gt)

    pred = sorted(pred, key=lambda x: x["conf"], reverse=True)

    # organize GT by image
    gt_per_image = {}
    for g in gt:
        img = g["image_name"]
        gt_per_image.setdefault(img, []).append(g)

    # GT matched flag
    gt_used = {}
    for img in gt_per_image:
        gt_used[img] = [False] * len(gt_per_image[img])

    tp = np.zeros(len(pred))
    fp = np.zeros(len(pred))

    for i, p in enumerate(pred):

        img = p["image_name"]

        if img not in gt_per_image:
            fp[i] = 1
            continue

        gts = gt_per_image[img]

        best_iou = 0
        best_gt_idx = -1

        for j, g in enumerate(gts):

            if gt_used[img][j]:
                continue

            iou = compute_iou(p["bbox"], g["bbox"])

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_thr:
            tp[i] = 1
            gt_used[img][best_gt_idx] = True
        else:
            fp[i] = 1

    return tp, fp, n_gt


def compute_precision_recall(tp, fp, n_gt):

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / (n_gt + 1e-16)
    precision = tp_cum / (tp_cum + fp_cum + 1e-16)

    return precision, recall


def compute_ap(precision, recall):
    """
    VOC/YOLO interpolated AP
    """

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return ap


def compute_map(gt_boxes, pred_boxes, num_classes, iou_thr=0.5):

    aps = []

    for c in range(num_classes):

        tp, fp, n_gt = evaluate_class(gt_boxes, pred_boxes, c, iou_thr)

        if n_gt == 0:
            aps.append(0)
            continue

        precision, recall = compute_precision_recall(tp, fp, n_gt)

        ap = compute_ap(precision, recall)

        aps.append(ap)

    mAP = np.mean(aps)

    return mAP, aps


if __name__ == "__main__":

    gt_boxes = [
        {"image_name": 0, "class": 0, "bbox": [10, 10, 50, 50]},
        {"image_name": 1, "class": 0, "bbox": [30, 30, 80, 80]},
        {"image_name": 1, "class": 1, "bbox": [100, 100, 150, 150]},
    ]

    pred_boxes = [
        {"image_name": 0, "class": 0, "bbox": [12, 12, 48, 48], "conf": 0.95},
        {"image_name": 1, "class": 0, "bbox": [35, 35, 75, 75], "conf": 0.90},
        {"image_name": 1, "class": 1, "bbox": [102, 102, 148, 148], "conf": 0.85},
        {"image_name": 1, "class": 0, "bbox": [0, 0, 20, 20], "conf": 0.60},
    ]

    mAP, APs = compute_map(gt_boxes, pred_boxes, num_classes=2)

    print("AP per class:", APs)
    print("mAP:", mAP)