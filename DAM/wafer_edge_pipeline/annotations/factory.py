from __future__ import annotations

from pathlib import Path

from wafer_edge_pipeline.annotations.loaders import (
    load_coco_instances,
    load_labelme_instances,
    load_mask_instances,
    load_yolo_bbox_instances,
)
from wafer_edge_pipeline.types import AnnotationInstance


def load_instances(
    annotation_format: str,
    image_path: Path,
    image_shape: tuple[int, int],
    annotation_dir: Path,
    class_names: tuple[str, ...],
    min_area: float,
    coco_index: dict | None = None,
) -> list[AnnotationInstance]:
    class_to_id = {
        name: index
        for index, name in enumerate(class_names)
    }

    if annotation_format == "none":
        return []

    if annotation_format == "yolo_bbox":
        return load_yolo_bbox_instances(
            annotation_dir / f"{image_path.stem}.txt",
            image_shape,
        )

    if annotation_format == "labelme":
        return load_labelme_instances(
            annotation_dir / f"{image_path.stem}.json",
            class_to_id,
        )

    if annotation_format == "mask":
        return load_mask_instances(
            annotation_dir / f"{image_path.stem}.png",
            image_shape,
            len(class_names),
            min_area,
        )

    if annotation_format == "coco":
        if coco_index is None:
            raise ValueError("COCO 模式缺少 coco_index")

        return load_coco_instances(
            image_path.name,
            coco_index,
            class_to_id,
        )

    raise ValueError(f"不支援標註格式：{annotation_format}")
