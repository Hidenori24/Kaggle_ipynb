"""Reconstructs a top-down occupancy heightmap for one container from the
`container_list` observation the environment provides every step.

Rather than trying to reverse-engineer the exact chamfered ULD mesh (the
open-cut-corner shape used for the LD3 profile) or parse the rendered depth
image, we rebuild the occupancy state directly from the authoritative
`packed_items` positions/orientations supplied in the observation. This is
exact for every item our own agent has placed, and a safe axis-aligned
over-approximation for anything that arrived already packed (pre-filled
containers), which only ever makes us slightly more conservative.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import item_world_aabb


class ContainerState:
    def __init__(self, container: dict, grid_n: int = 32):
        self.pos_index = container.get("index", 0)
        self.length = float(container["length"])   # X
        self.width = float(container["width"])      # Y (depth, door at -Y)
        self.height = float(container["height"])    # Z
        self.thickness = float(container.get("thickness", 0.02))
        self.cut_x = float(container.get("cut_x", 0.0) or 0.0)
        self.cut_y = float(container.get("cut_y", 0.0) or 0.0)
        self.is_prioritized = bool(container.get("is_prioritized", False))
        self.has_shelf = bool(container.get("shelf", False))
        # local x = world x - offset_x; offset_x equals the container's world
        # center x since local (0, 0, h/2+buffer) maps to `center` (see
        # Container.local_to_global / create in the simulator source).
        center = container.get("center", (0.0, 0.0, 0.0))
        self.offset_x = float(center[0])

        wall_margin = 0.005
        self.x_min = -self.length / 2.0 + self.thickness + wall_margin
        self.x_max = self.length / 2.0 - self.thickness - wall_margin
        self.y_min = -self.width / 2.0 + self.thickness + wall_margin
        self.y_max = self.width / 2.0 - self.thickness - wall_margin
        self.floor_z = self.thickness

        top_margin = 0.02
        if self.has_shelf:
            # Conservative simplification: treat the internal shelf plane as
            # a hard ceiling rather than reconstructing the exact partitioned
            # sub-volumes above/behind it. Trades a little capacity for
            # guaranteed collision safety.
            self.ceiling_z = self.height / 2.0 + self.thickness / 2.0 - top_margin
        else:
            self.ceiling_z = self.height - self.thickness - top_margin
        self.ceiling_z = max(self.ceiling_z, self.floor_z + 0.05)

        self.grid_n = grid_n
        self.cell_w = (self.x_max - self.x_min) / grid_n
        self.cell_h = (self.y_max - self.y_min) / grid_n
        self.height_grid = np.full((grid_n, grid_n), self.floor_z, dtype=np.float64)
        self.top_soft = np.zeros((grid_n, grid_n), dtype=bool)
        self.top_prioritized = np.zeros((grid_n, grid_n), dtype=bool)

        self._apply_cut_corner_keepout()
        self._build_from_packed_items(container.get("packed_items", []) or [])

    def local_x(self, world_x: float) -> float:
        return world_x - self.offset_x

    def to_world(self, local_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        return (local_xyz[0] + self.offset_x, local_xyz[1], local_xyz[2])

    def _apply_cut_corner_keepout(self) -> None:
        """The LD3 profile chamfers one bottom corner across the full depth
        of the container. We don't know a priori which x-side it's on, so we
        conservatively reserve both x-extremes up to the chamfer height.
        """
        if self.cut_x <= 0 or self.cut_y <= 0:
            return
        band_cells = max(1, int(math.ceil(self.cut_x / max(self.cell_w, 1e-6))))
        band_cells = min(band_cells, self.grid_n // 2)
        cut_top = self.floor_z + self.cut_y
        self.height_grid[:band_cells, :] = np.maximum(self.height_grid[:band_cells, :], cut_top)
        self.height_grid[self.grid_n - band_cells:, :] = np.maximum(
            self.height_grid[self.grid_n - band_cells:, :], cut_top
        )

    def _grid_index_range(self, x0: float, x1: float, y0: float, y1: float):
        ix0 = int(math.floor((x0 - self.x_min) / self.cell_w))
        ix1 = int(math.ceil((x1 - self.x_min) / self.cell_w))
        iy0 = int(math.floor((y0 - self.y_min) / self.cell_h))
        iy1 = int(math.ceil((y1 - self.y_min) / self.cell_h))
        ix0 = max(0, min(ix0, self.grid_n - 1))
        iy0 = max(0, min(iy0, self.grid_n - 1))
        ix1 = max(ix0 + 1, min(ix1, self.grid_n))
        iy1 = max(iy0 + 1, min(iy1, self.grid_n))
        return ix0, ix1, iy0, iy1

    def _build_from_packed_items(self, packed_items: list[dict]) -> None:
        for item in packed_items:
            pos = item.get("pos")
            orn = item.get("orn")
            if pos is None or orn is None:
                continue
            lo, hi = item_world_aabb(pos, orn, item["length"], item["width"], item["height"])
            x0, x1 = self.local_x(lo[0]), self.local_x(hi[0])
            y0, y1 = lo[1], hi[1]
            top_z = float(hi[2])

            ix0, ix1, iy0, iy1 = self._grid_index_range(x0, x1, y0, y1)
            sub = self.height_grid[ix0:ix1, iy0:iy1]
            mask = top_z > sub
            if not mask.any():
                continue
            sub[mask] = top_z
            self.top_soft[ix0:ix1, iy0:iy1][mask] = bool(item.get("is_soft", False))
            self.top_prioritized[ix0:ix1, iy0:iy1][mask] = bool(item.get("is_prioritized", False))
