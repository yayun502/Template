from __future__ import annotations

import cv2
import numpy as np

from wafer_edge_pipeline.types import AnnotationInstance


def unwrap_image(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def circular_crop(
    array: np.ndarray,
    start_x: int,
    patch_width: int,
) -> np.ndarray:
    width = array.shape[1]
    indices = (
        np.arange(start_x, start_x + patch_width) % width
    ).astype(int)

    return array[:, indices].copy()


def unwrap_instance_to_mask(
    instance: AnnotationInstance,
    image_shape: tuple[int, int],
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    source_mask = np.zeros(image_shape, dtype=np.uint8)

    polygon = np.round(instance.polygon).astype(np.int32)
    cv2.fillPoly(source_mask, [polygon], color=255)

    return cv2.remap(
        source_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
