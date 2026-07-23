# Overlap Patch 重複預測處理：初版修改說明

建議在初版 `map_predictions.py` 做四個主要修改。NMS 應放在**完整 unwrap 全域座標**，不要等映射成原圖曲線後才做。

# 1. 在 `Config` 加入 NMS 設定

找到：

```python
@dataclass
class Config:
```

在最後加入：

```python
    # 是否進行跨 patch 去重
    enable_global_nms: bool = True

    # 同類別 bbox 的 IoU 超過此值，視為重複
    global_nms_iou: float = 0.5

    # 一大一小框可使用 IoS 判定
    use_ios: bool = True
    global_nms_ios: float = 0.8

    # NMS 前先排除低信心預測
    min_confidence: float = 0.0
```

`config.yaml` 也加入：

```yaml
enable_global_nms: true
global_nms_iou: 0.5
use_ios: true
global_nms_ios: 0.8
min_confidence: 0.0
```

# 2. 新增完整 unwrap bbox 資料結構

放在 `Prediction` dataclass 後面：

```python
@dataclass
class GlobalDetection:
    """Detection 在完整 unwrap 圖中的位置。"""

    class_id: int
    confidence: float
    patch_index: int
    prediction_file: str
    x1: float
    y1: float
    x2: float
    y2: float
    prediction: Prediction
```

`x1`、`x2` 暫時不取 modulo，允許超過 `unwrap_width`，方便處理跨越 0°／360° 的 patch。

# 3. 將 patch bbox 轉成完整 unwrap bbox

```python
def prediction_to_global_detection(
    prediction: Prediction,
    patch_index: int,
    prediction_file: str,
    cfg: Config,
) -> GlobalDetection:
    local_x1, local_y1, local_x2, local_y2 = (
        yolo_bbox_to_patch_xyxy(
            prediction=prediction,
            patch_width=cfg.patch_width,
            patch_height=cfg.unwrap_height,
        )
    )

    start_x = patch_index * cfg.stride

    return GlobalDetection(
        class_id=prediction.class_id,
        confidence=prediction.confidence,
        patch_index=patch_index,
        prediction_file=prediction_file,
        x1=float(start_x + local_x1),
        y1=float(local_y1),
        x2=float(start_x + local_x2),
        y2=float(local_y2),
        prediction=prediction,
    )
```

# 4. 新增 IoU、IoS 與 circular NMS

一般 overlap metric：

```python
def box_overlap_metrics(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection
    smaller_area = min(area_a, area_b)

    iou = intersection / union if union > 0 else 0.0
    ios = intersection / smaller_area if smaller_area > 0 else 0.0
    return iou, ios
```

Circular 比較：

```python
def circular_overlap_metrics(
    detection_a: GlobalDetection,
    detection_b: GlobalDetection,
    unwrap_width: int,
):
    box_a = (
        detection_a.x1,
        detection_a.y1,
        detection_a.x2,
        detection_a.y2,
    )

    best_iou = 0.0
    best_ios = 0.0

    for shift in (-unwrap_width, 0, unwrap_width):
        box_b = (
            detection_b.x1 + shift,
            detection_b.y1,
            detection_b.x2 + shift,
            detection_b.y2,
        )

        iou, ios = box_overlap_metrics(box_a, box_b)
        best_iou = max(best_iou, iou)
        best_ios = max(best_ios, ios)

    return best_iou, best_ios
```

Class-wise circular NMS：

```python
def classwise_circular_nms(
    detections: list[GlobalDetection],
    cfg: Config,
) -> list[GlobalDetection]:
    kept = []

    for class_id in sorted({d.class_id for d in detections}):
        current = [d for d in detections if d.class_id == class_id]
        current.sort(key=lambda d: d.confidence, reverse=True)

        while current:
            best = current.pop(0)
            kept.append(best)
            remaining = []

            for candidate in current:
                iou, ios = circular_overlap_metrics(
                    best,
                    candidate,
                    cfg.unwrap_width,
                )

                duplicate = iou >= cfg.global_nms_iou
                if cfg.use_ios:
                    duplicate = duplicate or ios >= cfg.global_nms_ios

                if not duplicate:
                    remaining.append(candidate)

            current = remaining

    return kept
```

# 5. 修改 `process_one_wafer()`

舊版是讀到 prediction 後立刻映射與畫圖。新版需改成三個階段。

## 階段一：收集全部 patch prediction

```python
all_detections = []

for prediction_file in sorted(prediction_files):
    _, patch_index = parse_patch_name(prediction_file)

    for prediction in load_prediction_txt(prediction_file):
        if prediction.confidence < cfg.min_confidence:
            continue

        all_detections.append(
            prediction_to_global_detection(
                prediction=prediction,
                patch_index=patch_index,
                prediction_file=prediction_file.name,
                cfg=cfg,
            )
        )
```

## 階段二：跨 patch 去重

```python
if cfg.enable_global_nms:
    final_detections = classwise_circular_nms(
        all_detections,
        cfg,
    )
else:
    final_detections = all_detections
```

## 階段三：NMS 後才映射回原圖

```python
for detection in final_detections:
    polygon = global_detection_to_original_polygon(
        detection=detection,
        cfg=cfg,
    )

    draw_polygon_prediction(
        canvas=canvas,
        polygon=polygon,
        prediction=detection.prediction,
        cfg=cfg,
    )
```

JSON 建議記錄：

```python
{
    "raw_prediction_count": len(all_detections),
    "final_prediction_count": len(final_detections),
    "nms_enabled": cfg.enable_global_nms,
    "nms_iou": cfg.global_nms_iou,
    "nms_ios": cfg.global_nms_ios,
    "predictions": output_predictions,
}
```

Terminal 顯示：

```python
print(
    f"[完成] {source_stem}: "
    f"{len(all_detections)} → "
    f"{len(final_detections)}"
)
```

# Threshold 建議

```text
global_nms_iou = 0.3：去重積極
global_nms_iou = 0.5：建議起始值
global_nms_iou = 0.7：去重保守
```

若同一 defect 的一個框較大、另一個只框到一部分，IoU 可能不高，此時使用：

```text
global_nms_ios = 0.8
```

通常比直接把 IoU threshold 降得很低更安全。
