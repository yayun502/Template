from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class AnnotationInstance:
    class_id: int
    polygon: np.ndarray
    source_type: str
    instance_id: str | None = None

    def validate(self) -> None:
        if self.class_id < 0:
            raise ValueError("class_id 不可小於 0")

        if self.polygon.ndim != 2 or self.polygon.shape[1] != 2:
            raise ValueError("polygon 必須是 Nx2")

        if len(self.polygon) < 3:
            raise ValueError("polygon 至少要有 3 個點")


@dataclass
class GeometryInfo:
    center_x: float
    center_y: float
    inner_radius: float
    outer_radius: float


TaskType = Literal["detect", "segment"]
