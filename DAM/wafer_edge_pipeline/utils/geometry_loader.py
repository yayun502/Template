from __future__ import annotations

import json
from pathlib import Path

from wafer_edge_pipeline.config import PipelineConfig
from wafer_edge_pipeline.types import GeometryInfo


def load_geometry_table(path: str | None) -> dict:
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_geometry(
    image_path: Path,
    cfg: PipelineConfig,
    table: dict,
) -> GeometryInfo:
    item = table.get(image_path.name)
    if item is None:
        item = table.get(image_path.stem)

    if item is None:
        return GeometryInfo(
            center_x=cfg.center_x,
            center_y=cfg.center_y,
            inner_radius=cfg.inner_radius,
            outer_radius=cfg.outer_radius,
        )

    center_x = float(item["center_x"])
    center_y = float(item["center_y"])

    if "inner_radius" in item and "outer_radius" in item:
        inner_radius = float(item["inner_radius"])
        outer_radius = float(item["outer_radius"])
    else:
        wafer_radius = float(item["wafer_radius"])
        inner_radius = (
            wafer_radius
            - float(item.get("inner_margin", 200))
        )
        outer_radius = (
            wafer_radius
            + float(item.get("outer_margin", 20))
        )

    return GeometryInfo(
        center_x=center_x,
        center_y=center_y,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )
