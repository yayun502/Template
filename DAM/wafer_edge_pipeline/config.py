from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    input_root: str = "input_dataset"
    output_root: str = "wafer_edge_dataset"

    annotation_format: str = "labelme"
    task: str = "segment"

    names: tuple[str, ...] = (
        "Lowfill",
        "Overfill",
        "Splash",
    )

    center_x: float = 2048.0
    center_y: float = 2048.0
    inner_radius: float = 1800.0
    outer_radius: float = 2020.0
    center_json: str | None = None

    unwrap_width: int = 8192
    unwrap_height: int = 256
    patch_width: int = 512
    stride: int = 384

    min_area: float = 8.0
    polygon_epsilon_ratio: float = 0.002

    negative_keep_ratio: float = 1.0
    seed: int = 42

    save_debug_images: bool = True
    debug_max_per_split: int = 20

    coco_train_json: str = "annotations/train.json"
    coco_val_json: str = "annotations/val.json"
    coco_test_json: str = "annotations/test.json"

    def validate(self) -> None:
        if self.annotation_format not in {
            "mask",
            "yolo_bbox",
            "labelme",
            "coco",
            "none",
        }:
            raise ValueError("不支援的 annotation_format")

        if self.task not in {"detect", "segment"}:
            raise ValueError("task 必須是 detect 或 segment")

        if not self.names:
            raise ValueError("names 不可為空")

        if self.patch_width > self.unwrap_width:
            raise ValueError("patch_width 不可大於 unwrap_width")

        if self.stride <= 0:
            raise ValueError("stride 必須大於 0")

        if not 0.0 <= self.negative_keep_ratio <= 1.0:
            raise ValueError("negative_keep_ratio 必須介於 0 到 1")


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if "names" in raw:
        raw["names"] = tuple(raw["names"])

    cfg = PipelineConfig(**raw)
    cfg.validate()
    return cfg
