"""
infer.py

1. 將單張 wafer 展開並切 patch
2. 對每個 patch 做 YOLO 推論
3. 將 segmentation polygon 或 detection box 映射回原圖
4. 輸出視覺化結果與 JSON

注意：
- overlap patch 可能產生重複預測。
- 此範例使用「原圖座標 bounding box NMS」合併重複結果。
- segmentation 會保留映射回原圖的 polygon。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from preprocess import (
    build_polar_maps,
    circular_crop,
    load_config,
    unwrap_image,
)


def patch_xy_to_original(
    x: np.ndarray,
    y: np.ndarray,
    start_x: int,
    patch_width: int,
    unwrap_width: int,
    unwrap_height: int,
    center_x: float,
    center_y: float,
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    """將 patch 座標轉回原始 wafer 座標。"""
    # patch x 加回長條圖 offset，並處理 360° 接縫。
    global_u = (x + start_x) % unwrap_width

    theta = 2.0 * np.pi * global_u / unwrap_width

    # y=0 對應 inner radius，底部對應 outer radius。
    radius = inner_radius + (
        y / max(unwrap_height - 1, 1)
    ) * (outer_radius - inner_radius)

    original_x = center_x + radius * np.cos(theta)
    original_y = center_y + radius * np.sin(theta)

    return np.stack([original_x, original_y], axis=1)


def polygon_bbox(points: np.ndarray) -> list[float]:
    """取得 polygon 的 xyxy bounding box。"""
    return [
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    ]


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    """計算兩個 xyxy box 的 IoU。"""
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

    return intersection / union if union > 0 else 0.0


def classwise_nms(
    detections: list[dict],
    iou_threshold: float,
) -> list[dict]:
    """依類別執行簡單 NMS。"""
    kept: list[dict] = []

    for class_id in sorted({d["class_id"] for d in detections}):
        current = [d for d in detections if d["class_id"] == class_id]
        current.sort(key=lambda d: d["confidence"], reverse=True)

        while current:
            best = current.pop(0)
            kept.append(best)

            current = [
                d for d in current
                if box_iou(best["bbox"], d["bbox"]) < iou_threshold
            ]

    return kept


def draw_results(
    image: np.ndarray,
    detections: list[dict],
    class_names: dict[int, str],
) -> np.ndarray:
    """畫出映射回原圖的結果。"""
    canvas = image.copy()

    for det in detections:
        points = np.asarray(det["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], isClosed=True, color=(0, 0, 255), thickness=3)

        x1, y1, _, _ = det["bbox"]
        label = (
            f"{class_names.get(det['class_id'], str(det['class_id']))} "
            f"{det['confidence']:.2f}"
        )
        cv2.putText(
            canvas,
            label,
            (int(x1), max(20, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="inference_output")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    cfg = load_config(args.config)

    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"無法讀取影像：{image_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 單張推論範例先使用 config 中的固定幾何資料。
    cx = cfg.center_x
    cy = cfg.center_y
    inner_radius = cfg.inner_radius
    outer_radius = cfg.outer_radius

    map_x, map_y = build_polar_maps(
        center=(cx, cy),
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        output_width=cfg.unwrap_width,
        output_height=cfg.unwrap_height,
    )
    unwrapped = unwrap_image(image, map_x, map_y, is_mask=False)

    model = YOLO(args.model)
    detections: list[dict] = []

    for patch_index, start_x in enumerate(range(0, cfg.unwrap_width, cfg.stride)):
        patch = circular_crop(
            unwrapped,
            start_x=start_x,
            patch_width=cfg.patch_width,
        )

        result = model.predict(
            source=patch,
            conf=args.conf,
            iou=args.iou,
            imgsz=max(cfg.patch_width, cfg.unwrap_height),
            device=args.device,
            verbose=False,
        )[0]

        # Segmentation 模型
        if result.masks is not None and result.boxes is not None:
            polygons = result.masks.xy

            for obj_index, polygon in enumerate(polygons):
                if len(polygon) < 3:
                    continue

                class_id = int(result.boxes.cls[obj_index].item())
                confidence = float(result.boxes.conf[obj_index].item())

                polygon = np.asarray(polygon, dtype=np.float32)
                original_polygon = patch_xy_to_original(
                    x=polygon[:, 0],
                    y=polygon[:, 1],
                    start_x=start_x,
                    patch_width=cfg.patch_width,
                    unwrap_width=cfg.unwrap_width,
                    unwrap_height=cfg.unwrap_height,
                    center_x=cx,
                    center_y=cy,
                    inner_radius=inner_radius,
                    outer_radius=outer_radius,
                )

                detections.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "polygon": original_polygon.tolist(),
                    "bbox": polygon_bbox(original_polygon),
                    "patch_index": patch_index,
                })

        # Detection 模型
        elif result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(np.float32)

                # 取矩形四邊多個點，映射後會形成彎曲帶狀 polygon。
                samples = 20
                top_x = np.linspace(x1, x2, samples)
                top_y = np.full(samples, y1)
                right_x = np.full(samples, x2)
                right_y = np.linspace(y1, y2, samples)
                bottom_x = np.linspace(x2, x1, samples)
                bottom_y = np.full(samples, y2)
                left_x = np.full(samples, x1)
                left_y = np.linspace(y2, y1, samples)

                px = np.concatenate([top_x, right_x, bottom_x, left_x])
                py = np.concatenate([top_y, right_y, bottom_y, left_y])

                original_polygon = patch_xy_to_original(
                    x=px,
                    y=py,
                    start_x=start_x,
                    patch_width=cfg.patch_width,
                    unwrap_width=cfg.unwrap_width,
                    unwrap_height=cfg.unwrap_height,
                    center_x=cx,
                    center_y=cy,
                    inner_radius=inner_radius,
                    outer_radius=outer_radius,
                )

                detections.append({
                    "class_id": int(box.cls.item()),
                    "confidence": float(box.conf.item()),
                    "polygon": original_polygon.tolist(),
                    "bbox": polygon_bbox(original_polygon),
                    "patch_index": patch_index,
                })

    merged = classwise_nms(detections, iou_threshold=args.iou)

    class_names = model.names
    visualization = draw_results(image, merged, class_names)

    out_image = output_dir / f"{image_path.stem}_result.png"
    out_json = output_dir / f"{image_path.stem}_result.json"
    out_unwrap = output_dir / f"{image_path.stem}_unwrapped.png"

    cv2.imwrite(str(out_image), visualization)
    cv2.imwrite(str(out_unwrap), unwrapped)
    out_json.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"原始候選數：{len(detections)}")
    print(f"NMS 後數量：{len(merged)}")
    print(f"結果影像：{out_image}")
    print(f"結果 JSON：{out_json}")


if __name__ == "__main__":
    main()
