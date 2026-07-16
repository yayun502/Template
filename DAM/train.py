"""
train.py

使用 Ultralytics YOLO 訓練。
模型名稱請依公司環境中已下載的權重修改。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="wafer_edge_dataset/dataset.yaml")
    parser.add_argument("--model", default="yolov8m-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/wafer_edge")
    parser.add_argument("--name", default="train")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"找不到 dataset yaml：{args.data}")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,

        # 展開後的上下方向具有物理意義，不做上下翻轉。
        flipud=0.0,

        # 左右翻轉代表圓周方向反轉；通常可以使用。
        fliplr=0.5,

        # 展開後方向已固定，不建議做大角度旋轉。
        degrees=0.0,
        perspective=0.0,

        # 點膠是連續結構，Mosaic 可能製造不自然接縫。
        mosaic=0.0,

        # 輕微亮度與飽和度變化。
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,

        # 保留長寬比並 padding。
        rect=True,

        patience=30,
        save=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
