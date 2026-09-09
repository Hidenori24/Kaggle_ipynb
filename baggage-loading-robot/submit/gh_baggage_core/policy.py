"""Online (sequential) placement policy.

For every visible pool item x every orientation x every container, we ask
the heightmap packer (packing.best_position) for its best landing spot, then
pick the single (item, orientation, container, position) combination with
the lowest overall cost. Costs combine:

  - resulting stack height (lower is better: keeps things low & stable)
  - a large penalty for placing a prioritized item into a non-prioritized
    container when a prioritized container exists (scoring rule)
  - a small penalty for "spending" the prioritized container's space on a
    non-prioritized item when another container is available (soft
    reservation, not a hard rule -- there is no scoring penalty for this,
    but it reduces the chance of the prioritized container filling up
    before priority bags arrive)
  - a small penalty when the chosen anchor forces a priority/soft item to
    be buried under (or to sit under) an incompatible item, since that is
    exactly what the placement/soft-item scores penalize
  - a small bonus for a larger, flatter base of support (stability)
"""
from __future__ import annotations

import numpy as np

from .container_state import ContainerState
from .geometry import NUM_ORIENTATIONS, oriented_dims
from .packing import best_position

PRIORITY_CONTAINER_VIOLATION_PENALTY = 1000.0
PRIORITY_CONTAINER_RESERVE_PENALTY = 0.05
TOP_CONFLICT_PENALTY = 0.5
FOOTPRINT_BONUS_SCALE = 0.001


class Policy:
    def __init__(self, lookahead_k: int | None = None):
        self.lookahead_k = lookahead_k

    def act(self, observation: dict) -> dict:
        try:
            return self._act(observation)
        except Exception:
            return self._fallback_action(observation)

    def _act(self, observation: dict) -> dict:
        container_list = observation.get("container_list") or []
        pool_list = observation.get("pool_list") or []

        if not pool_list or not container_list:
            return self._fallback_action(observation)

        states = [ContainerState(c) for c in container_list]
        has_priority_container = any(s.is_prioritized for s in states)
        multi_container = len(states) > 1

        best = None  # (score, item_pos_idx, container_pos_idx, orn_idx, pos_result, dims)

        for item_pos_idx, item in enumerate(pool_list):
            length, width, height = item["length"], item["width"], item["height"]
            is_soft = bool(item.get("is_soft", False))
            is_prioritized = bool(item.get("is_prioritized", False))

            for orn_idx in range(NUM_ORIENTATIONS):
                dl, dw, dh = oriented_dims(length, width, height, orn_idx)

                for c_idx, state in enumerate(states):
                    container_penalty = 0.0
                    if is_prioritized and has_priority_container and not state.is_prioritized:
                        container_penalty += PRIORITY_CONTAINER_VIOLATION_PENALTY
                    if (
                        not is_prioritized
                        and has_priority_container
                        and state.is_prioritized
                        and multi_container
                    ):
                        container_penalty += PRIORITY_CONTAINER_RESERVE_PENALTY

                    result = best_position(
                        state, dl, dw, dh,
                        avoid_soft_top=not is_soft,
                        avoid_priority_top=not is_prioritized,
                    )
                    if result is None:
                        continue

                    score = result["top"] + container_penalty
                    if result["conflict"]:
                        score += TOP_CONFLICT_PENALTY
                    score -= FOOTPRINT_BONUS_SCALE * (dl * dw)

                    if best is None or score < best[0]:
                        best = (score, item_pos_idx, c_idx, orn_idx, result, (dl, dw, dh))

        if best is None:
            return self._fallback_action(observation)

        _, item_pos_idx, c_idx, orn_idx, result, _dims = best
        # `place_pos` is the container-relative local coordinate the env
        # expects (see GroundHandlingEnv.step -> Container.local_to_global).
        place_pos = np.array([result["x"], result["y"], result["z"]], dtype=np.float32)

        return {
            "item_idx": item_pos_idx,
            "container_idx": c_idx,
            "place_pos": place_pos,
            "orientation": orn_idx,
        }

    @staticmethod
    def _fallback_action(observation: dict) -> dict:
        container_list = observation.get("container_list") or []
        pool_list = observation.get("pool_list") or []
        if container_list:
            c = container_list[0]
            thickness = float(c.get("thickness", 0.02))
            length = float(c.get("length", 1.0))
            width = float(c.get("width", 1.0))
            item_h = 0.2
            if pool_list:
                item_h = float(pool_list[0].get("height", 0.2))
            x = -length / 2.0 + thickness + 0.1
            y = -width / 2.0 + thickness + 0.1
            z = thickness + item_h / 2.0
            place_pos = np.array([x, y, z], dtype=np.float32)
        else:
            place_pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        return {
            "item_idx": 0,
            "container_idx": 0,
            "place_pos": place_pos,
            "orientation": 0,
        }
