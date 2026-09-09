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
    def __init__(self, container: dict, grid_n: int = 40):
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

        # The validator's `check_inclusion` requires every wall-plane dot
        # product to be <= inclusion_margin (-0.005 m in the evaluation
        # config): landing exactly on a geometric boundary (dot == 0) fails.
        # We keep every hard container-wall boundary at least SAFETY_MARGIN
        # clear of the true wall so this can never happen, even with
        # floating point noise.
        SAFETY_MARGIN = 0.012
        self.x_min = -self.length / 2.0 + self.thickness + SAFETY_MARGIN
        self.x_max = self.length / 2.0 - self.thickness - SAFETY_MARGIN
        self.y_min = -self.width / 2.0 + self.thickness + SAFETY_MARGIN
        self.y_max = self.width / 2.0 - self.thickness - SAFETY_MARGIN
        self.floor_z = self.thickness + SAFETY_MARGIN

        top_margin = 0.02
        self.ceiling_z = self.height - self.thickness - top_margin
        self.ceiling_z = max(self.ceiling_z, self.floor_z + 0.05)

        self.grid_n = grid_n
        self.cell_w = (self.x_max - self.x_min) / grid_n
        self.cell_h = (self.y_max - self.y_min) / grid_n
        self.height_grid = np.full((grid_n, grid_n), self.floor_z, dtype=np.float64)
        self.ceiling_grid = np.full((grid_n, grid_n), self.ceiling_z, dtype=np.float64)
        self.top_soft = np.zeros((grid_n, grid_n), dtype=bool)
        self.top_prioritized = np.zeros((grid_n, grid_n), dtype=bool)
        self.corner_keepout = np.zeros((grid_n, grid_n), dtype=bool)

        if self.has_shelf:
            self._apply_shelf_ceiling()
        self._apply_cut_corner_keepout()
        self._build_from_packed_items(container.get("packed_items", []) or [])

    def _apply_shelf_ceiling(self) -> None:
        """Conservative simplification: treat the internal shelf plane as a
        hard ceiling rather than reconstructing the exact partitioned
        sub-volumes above/behind it. The shelf plank spans roughly
        [height/2 + buffer, height/2 + thickness + buffer] in Z (see
        Container._create_shelf in the simulator source) and only the local
        y >= ~0 half of the depth (it's built at y-center = width/4 with a
        half-extent of width/4 - thickness, i.e. y in
        [thickness, width/2 - thickness]) -- so capping the *entire*
        container at half-height would sacrifice an entire usable half for a
        thin plank that only occupies part of it. `buffer` isn't part of the
        observation we receive, so we stay well clear of height/2 rather
        than of the plank's exact underside, and treat local y < 0 (the
        whole door-side half) as shelf-free.
        """
        shelf_ceiling = max(self.height / 2.0 - 0.05, self.floor_z + 0.05)
        iy0 = self._y_index(0.0)
        self.ceiling_grid[:, iy0:] = np.minimum(self.ceiling_grid[:, iy0:], shelf_ceiling)

    def _y_index(self, y: float) -> int:
        idx = int(math.floor((y - self.y_min) / self.cell_h))
        return max(0, min(idx, self.grid_n))

    def local_x(self, world_x: float) -> float:
        return world_x - self.offset_x

    def to_world(self, local_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        return (local_xyz[0] + self.offset_x, local_xyz[1], local_xyz[2])

    def _apply_cut_corner_keepout(self) -> None:
        """The LD3 profile chamfers one bottom corner across the full depth
        of the container -- always on the local x_min side (see
        `write_open_cut_corner_cup_obj` / `Container._create_small_shelf` in
        the simulator source, which both anchor the cut at -length/2).

        Beyond the geometric cut itself, the validator's own transport-path
        check *forces* every item's entry X-coordinate to stay at least
        `cut_x` clear of x_min (`x_min = -length/2 + thickness + cut_x + ...`
        in `PlacementValidator.check_transport_path`), regardless of where it
        actually lands. So a target inside that band gets dragged sideways
        through the main interior at a raised height before doubling back --
        which can clip a nearby stack we'd otherwise consider perfectly
        clear. We mark this band `corner_keepout` so the search can strongly
        prefer landing spots outside of it, without banning it outright (it
        is still real, usable volume once nothing else is available).

        The same band also always hosts a physical "small shelf" ledge
        around mid-height (`Container._create_small_shelf` runs whether or
        not `require_shelf` is set), which we don't otherwise model at all.
        Rather than track its thin, hard-to-pin-down real position exactly,
        we treat the whole band as fully unusable (not just discouraged, as
        `in_corner_keepout` scoring alone would give it) -- it's a narrow
        strip that's already avoided almost everywhere else, so sacrificing
        its low chamfer pocket too costs little fill capacity in exchange
        for closing off a collision we otherwise have no way to see coming.
        """
        if self.cut_x <= 0 or self.cut_y <= 0:
            return
        band_cells = max(1, int(math.ceil(self.cut_x / max(self.cell_w, 1e-6))))
        band_cells = min(band_cells, self.grid_n - 1)
        self.height_grid[:band_cells, :] = self.ceiling_z
        self.corner_keepout[:band_cells, :] = True

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

    def _mark_occupied(self, x0: float, x1: float, y0: float, y1: float, top_z: float,
                        is_soft: bool, is_prioritized: bool) -> None:
        ix0, ix1, iy0, iy1 = self._grid_index_range(x0, x1, y0, y1)
        sub = self.height_grid[ix0:ix1, iy0:iy1]
        mask = top_z > sub
        if not mask.any():
            return
        sub[mask] = top_z
        self.top_soft[ix0:ix1, iy0:iy1][mask] = bool(is_soft)
        self.top_prioritized[ix0:ix1, iy0:iy1][mask] = bool(is_prioritized)

    def _build_from_packed_items(self, packed_items: list[dict]) -> None:
        # Our grid is coarse (a few cm per cell) relative to the validator's
        # own safety_margin (1.5cm in the evaluation config): a gap that
        # looks clear at grid resolution can still be a real-world graze.
        # Padding every occupied footprint by a bit more than that margin
        # keeps new placements from being planned into gaps the physics
        # engine would reject.
        pad = 0.03
        for item in packed_items:
            pos = item.get("pos")
            orn = item.get("orn")
            if pos is None or orn is None:
                continue
            lo, hi = item_world_aabb(pos, orn, item["length"], item["width"], item["height"])
            self._mark_occupied(
                self.local_x(lo[0]) - pad, self.local_x(hi[0]) + pad,
                lo[1] - pad, hi[1] + pad,
                float(hi[2]), item.get("is_soft", False), item.get("is_prioritized", False),
            )

    def place_virtual(self, x_center: float, y_center: float, dl: float, dw: float, top_z: float,
                       is_soft: bool, is_prioritized: bool) -> None:
        """Record a placement decision that hasn't actually happened in the
        simulator (no physics settling to read back) -- used by the offline
        planner's dry-run pack, where we choose our own coordinates for
        every item up front instead of reading them from `packed_items`.
        Uses the same conservative padding as real placements so the dry
        run doesn't plan into gaps tighter than a real one would allow.
        """
        pad = 0.03
        self._mark_occupied(
            x_center - dl / 2.0 - pad, x_center + dl / 2.0 + pad,
            y_center - dw / 2.0 - pad, y_center + dw / 2.0 + pad,
            top_z, is_soft, is_prioritized,
        )
