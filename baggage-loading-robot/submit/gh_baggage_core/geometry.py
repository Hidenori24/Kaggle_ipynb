"""Orientation / rotation helpers matching the simulator's ORNS convention.

`orientation` (0-5) rotates an item from its native (length, width, height)
box. The dimension permutation below mirrors
``simulator/src/ground_handling/utils.py::get_half_ext`` exactly (doubled).
"""
from __future__ import annotations

import numpy as np

NUM_ORIENTATIONS = 6


def oriented_dims(length: float, width: float, height: float, orn_idx: int) -> tuple[float, float, float]:
    """Return the (dx, dy, dz) footprint/height of an item under orn_idx."""
    if orn_idx == 0:
        return length, width, height
    if orn_idx == 1:
        return length, height, width
    if orn_idx == 2:
        return height, width, length
    if orn_idx == 3:
        return width, length, height
    if orn_idx == 4:
        return width, height, length
    if orn_idx == 5:
        return height, length, width
    raise ValueError(f"invalid orientation index: {orn_idx}")


def quat_rotate(vectors: np.ndarray, quat_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    """Rotate an (N, 3) array of local vectors by a PyBullet-style (x, y, z, w) quaternion."""
    x, y, z, w = quat_xyzw
    # Standard quaternion-to-rotation-matrix (row-major, applies to column vectors).
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return vectors @ r.T


def item_world_aabb(pos: tuple[float, float, float], orn: tuple[float, float, float, float],
                     length: float, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box (world frame) of a possibly-rotated box item."""
    hl, hw, hh = length / 2.0, width / 2.0, height / 2.0
    corners = np.array([
        [sx * hl, sy * hw, sz * hh]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])
    rotated = quat_rotate(corners, orn) + np.asarray(pos)
    return rotated.min(axis=0), rotated.max(axis=0)
