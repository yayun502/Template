from __future__ import annotations

import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="wafer_edge_dataset/dataset.yaml",
    )
    parser.add_argument(
        "--model",
        default="yolov8m-seg.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--project",
        default="runs/wafer_edge",
    )
    parser.add_argument("--name", default="train")
    args = parser.parse_args()

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
        flipud=0.0,
        fliplr=0.5,
        degrees=0.0,
        perspective=0.0,
        mosaic=0.0,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        rect=True,
        patience=30,
        save=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
