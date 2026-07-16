from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from wafer_edge_pipeline.types import AnnotationInstance


def rectangle_to_polygon(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray:
    return np.asarray(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )


def mask_to_instances(
    mask: np.ndarray,
    class_count: int,
    min_area: float = 1.0,
) -> list[AnnotationInstance]:
    instances: list[AnnotationInstance] = []

    for mask_value in range(1, class_count + 1):
        binary = ((mask == mask_value).astype(np.uint8) * 255)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for index, contour in enumerate(contours):
            if cv2.contourArea(contour) < min_area:
                continue

            polygon = contour.reshape(-1, 2).astype(np.float32)
            if len(polygon) < 3:
                continue

            instances.append(
                AnnotationInstance(
                    class_id=mask_value - 1,
                    polygon=polygon,
                    source_type="mask",
                    instance_id=f"mask_{mask_value}_{index}",
                )
            )

    return instances


def read_gray_mask(
    path: Path,
    image_shape: tuple[int, int],
) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"無法讀取 mask：{path}")

    if mask.shape[:2] != image_shape:
        raise ValueError(
            f"Mask 尺寸不一致：{mask.shape[:2]} vs {image_shape}"
        )

    return mask
