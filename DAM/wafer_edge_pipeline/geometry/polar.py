from __future__ import annotations

import numpy as np

from wafer_edge_pipeline.types import GeometryInfo


def build_polar_maps(
    geometry: GeometryInfo,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        output_width,
        endpoint=False,
        dtype=np.float32,
    )

    radius = np.linspace(
        geometry.inner_radius,
        geometry.outer_radius,
        output_height,
        dtype=np.float32,
    )

    theta_grid, radius_grid = np.meshgrid(theta, radius)

    map_x = (
        geometry.center_x
        + radius_grid * np.cos(theta_grid)
    )
    map_y = (
        geometry.center_y
        + radius_grid * np.sin(theta_grid)
    )

    return map_x.astype(np.float32), map_y.astype(np.float32)


def original_points_to_unwrapped(
    points: np.ndarray,
    geometry: GeometryInfo,
    unwrap_width: int,
    unwrap_height: int,
) -> np.ndarray:
    dx = points[:, 0] - geometry.center_x
    dy = points[:, 1] - geometry.center_y

    theta = np.arctan2(dy, dx)
    theta = np.mod(theta, 2.0 * np.pi)

    radius = np.sqrt(dx * dx + dy * dy)

    u = theta / (2.0 * np.pi) * unwrap_width
    v = (
        (radius - geometry.inner_radius)
        / max(
            geometry.outer_radius - geometry.inner_radius,
            1e-6,
        )
        * (unwrap_height - 1)
    )

    return np.stack([u, v], axis=1).astype(np.float32)


def unwrapped_points_to_original(
    points: np.ndarray,
    geometry: GeometryInfo,
    unwrap_width: int,
    unwrap_height: int,
) -> np.ndarray:
    u = points[:, 0]
    v = points[:, 1]

    theta = 2.0 * np.pi * u / unwrap_width
    radius = (
        geometry.inner_radius
        + v / max(unwrap_height - 1, 1)
        * (
            geometry.outer_radius
            - geometry.inner_radius
        )
    )

    x = geometry.center_x + radius * np.cos(theta)
    y = geometry.center_y + radius * np.sin(theta)

    return np.stack([x, y], axis=1).astype(np.float32)
