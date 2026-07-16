from __future__ import annotations

import cv2
import numpy as np


def contour_to_segment_line(
    contour: np.ndarray,
    class_id: int,
    width: int,
    height: int,
    epsilon_ratio: float,
) -> str | None:
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(0.5, epsilon_ratio * perimeter)
    polygon = cv2.approxPolyDP(contour, epsilon, True)

    points = polygon.reshape(-1, 2)
    if len(points) < 3:
        return None

    values = [str(class_id)]

    for x, y in points:
        nx = np.clip(float(x) / width, 0.0, 1.0)
        ny = np.clip(float(y) / height, 0.0, 1.0)
        values.extend([f"{nx:.6f}", f"{ny:.6f}"])

    return " ".join(values)


def contour_to_detect_line(
    contour: np.ndarray,
    class_id: int,
    width: int,
    height: int,
) -> str:
    x, y, w, h = cv2.boundingRect(contour)

    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    nw = w / width
    nh = h / height

    return (
        f"{class_id} "
        f"{cx:.6f} {cy:.6f} "
        f"{nw:.6f} {nh:.6f}"
    )


def instance_patch_mask_to_yolo_lines(
    patch_mask: np.ndarray,
    class_id: int,
    task: str,
    min_area: float,
    epsilon_ratio: float,
) -> list[str]:
    lines: list[str] = []

    contours, _ = cv2.findContours(
        patch_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = patch_mask.shape[:2]

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        if task == "segment":
            line = contour_to_segment_line(
                contour,
                class_id,
                width,
                height,
                epsilon_ratio,
            )
            if line is not None:
                lines.append(line)

        elif task == "detect":
            lines.append(
                contour_to_detect_line(
                    contour,
                    class_id,
                    width,
                    height,
                )
            )
        else:
            raise ValueError("task 必須是 detect 或 segment")

    return lines
