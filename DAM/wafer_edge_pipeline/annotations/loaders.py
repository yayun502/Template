from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from wafer_edge_pipeline.annotations.common import (
    mask_to_instances,
    read_gray_mask,
    rectangle_to_polygon,
)
from wafer_edge_pipeline.types import AnnotationInstance


def load_yolo_bbox_instances(
    label_path: Path,
    image_shape: tuple[int, int],
) -> list[AnnotationInstance]:
    image_h, image_w = image_shape
    instances: list[AnnotationInstance] = []

    if not label_path.exists():
        return instances

    for line_no, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_no} 不是 YOLO bbox 格式"
            )

        class_id = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:])

        x1 = (cx - w / 2.0) * image_w
        y1 = (cy - h / 2.0) * image_h
        x2 = (cx + w / 2.0) * image_w
        y2 = (cy + h / 2.0) * image_h

        polygon = rectangle_to_polygon(x1, y1, x2, y2)

        instances.append(
            AnnotationInstance(
                class_id=class_id,
                polygon=polygon,
                source_type="yolo_bbox",
                instance_id=f"{label_path.stem}_{line_no}",
            )
        )

    return instances


def load_labelme_instances(
    json_path: Path,
    class_to_id: dict[str, int],
) -> list[AnnotationInstance]:
    if not json_path.exists():
        return []

    data = json.loads(json_path.read_text(encoding="utf-8"))
    instances: list[AnnotationInstance] = []

    for index, shape in enumerate(data.get("shapes", [])):
        label = shape.get("label")
        if label not in class_to_id:
            raise ValueError(
                f"{json_path.name} 中有未定義類別：{label}"
            )

        shape_type = shape.get("shape_type", "polygon")
        points = np.asarray(shape.get("points", []), dtype=np.float32)

        if shape_type == "rectangle":
            if len(points) != 2:
                continue
            x1, y1 = points[0]
            x2, y2 = points[1]
            polygon = rectangle_to_polygon(
                min(x1, x2),
                min(y1, y2),
                max(x1, x2),
                max(y1, y2),
            )

        elif shape_type == "circle":
            if len(points) != 2:
                continue

            center = points[0]
            edge = points[1]
            radius = float(np.linalg.norm(edge - center))

            angles = np.linspace(
                0.0,
                2.0 * np.pi,
                64,
                endpoint=False,
            )
            polygon = np.stack(
                [
                    center[0] + radius * np.cos(angles),
                    center[1] + radius * np.sin(angles),
                ],
                axis=1,
            ).astype(np.float32)

        else:
            polygon = points.astype(np.float32)

        if len(polygon) < 3:
            continue

        instances.append(
            AnnotationInstance(
                class_id=class_to_id[label],
                polygon=polygon,
                source_type="labelme",
                instance_id=f"{json_path.stem}_{index}",
            )
        )

    return instances


def load_coco_index(coco_json_path: Path) -> dict:
    data = json.loads(coco_json_path.read_text(encoding="utf-8"))

    categories = {
        int(item["id"]): item["name"]
        for item in data.get("categories", [])
    }

    image_id_by_name = {
        Path(item["file_name"]).name: int(item["id"])
        for item in data.get("images", [])
    }

    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in data.get("annotations", []):
        annotations_by_image.setdefault(
            int(annotation["image_id"]),
            [],
        ).append(annotation)

    return {
        "categories": categories,
        "image_id_by_name": image_id_by_name,
        "annotations_by_image": annotations_by_image,
    }


def load_coco_instances(
    image_name: str,
    coco_index: dict,
    class_to_id: dict[str, int],
) -> list[AnnotationInstance]:
    image_id = coco_index["image_id_by_name"].get(Path(image_name).name)
    if image_id is None:
        return []

    instances: list[AnnotationInstance] = []

    for ann in coco_index["annotations_by_image"].get(image_id, []):
        category_name = coco_index["categories"][int(ann["category_id"])]

        if category_name not in class_to_id:
            raise ValueError(
                f"COCO 類別未出現在 config.names：{category_name}"
            )

        segmentation = ann.get("segmentation")
        added = False

        if isinstance(segmentation, list) and segmentation:
            for part_index, segment in enumerate(segmentation):
                polygon = np.asarray(
                    segment,
                    dtype=np.float32,
                ).reshape(-1, 2)

                if len(polygon) < 3:
                    continue

                instances.append(
                    AnnotationInstance(
                        class_id=class_to_id[category_name],
                        polygon=polygon,
                        source_type="coco_polygon",
                        instance_id=f"{ann.get('id')}_{part_index}",
                    )
                )
                added = True

        if not added and "bbox" in ann:
            x, y, w, h = map(float, ann["bbox"])
            polygon = rectangle_to_polygon(
                x,
                y,
                x + w,
                y + h,
            )

            instances.append(
                AnnotationInstance(
                    class_id=class_to_id[category_name],
                    polygon=polygon,
                    source_type="coco_bbox",
                    instance_id=str(ann.get("id")),
                )
            )

    return instances


def load_mask_instances(
    mask_path: Path,
    image_shape: tuple[int, int],
    class_count: int,
    min_area: float,
) -> list[AnnotationInstance]:
    if not mask_path.exists():
        return []

    mask = read_gray_mask(mask_path, image_shape)
    return mask_to_instances(mask, class_count, min_area)
