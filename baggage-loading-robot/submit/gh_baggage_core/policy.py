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

import time

import numpy as np

from .container_state import ContainerState
from .geometry import NUM_ORIENTATIONS, oriented_dims
from .packing import best_position

PRIORITY_CONTAINER_VIOLATION_PENALTY = 1000.0
PRIORITY_CONTAINER_RESERVE_PENALTY = 0.05
TOP_CONFLICT_PENALTY = 0.5
PATH_BLOCKED_PENALTY = 100.0
UNSTABLE_PENALTY = 100.0
CORNER_KEEPOUT_PENALTY = 60.0
FOOTPRINT_BONUS_SCALE = 0.001

# The evaluation harness enforces an 8-10s wall-clock budget per policy()
# call and, on timeout, substitutes a *random* action of its own -- which is
# far more likely to fail the transport-path/settle checks and end the whole
# episode early than a merely-suboptimal choice of ours would be. We budget
# well under that limit so slower/loaded evaluation hardware can never push
# us over it, and degrade gracefully (return the best candidate found so
# far) rather than risk the harness's own fallback.
TIME_BUDGET_SECONDS = 4.0


class Policy:
    def __init__(self, lookahead_k: int | None = None):
        self.lookahead_k = lookahead_k

    def act(self, observation: dict) -> dict:
        try:
            return self._act(observation)
        except Exception:
            return self._fallback_action(observation)

    def _act(self, observation: dict) -> dict:
        deadline = time.perf_counter() + TIME_BUDGET_SECONDS
        container_list = observation.get("container_list") or []
        pool_list = observation.get("pool_list") or []

        if not pool_list or not container_list:
            return self._fallback_action(observation)

        states = [ContainerState(c) for c in container_list]
        has_priority_container = any(s.is_prioritized for s in states)
        multi_container = len(states) > 1

        # Evaluate the biggest items first: a good decision matters most for
        # them, so if the time budget forces an early cutoff we still end up
        # having considered the placements that matter most.
        item_order = sorted(
            range(len(pool_list)),
            key=lambda i: -(pool_list[i]["length"] * pool_list[i]["width"] * pool_list[i]["height"]),
        )

        best = None  # (score, item_pos_idx, container_pos_idx, orn_idx, pos_result, dims)

        for item_pos_idx in item_order:
            item = pool_list[item_pos_idx]
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
                    if result["path_blocked"]:
                        score += PATH_BLOCKED_PENALTY
                    if result["unstable"]:
                        score += UNSTABLE_PENALTY
                    if result["in_corner_keepout"]:
                        score += CORNER_KEEPOUT_PENALTY
                    score -= FOOTPRINT_BONUS_SCALE * (dl * dw)

                    if best is None or score < best[0]:
                        best = (score, item_pos_idx, c_idx, orn_idx, result, (dl, dw, dh))

            if time.perf_counter() > deadline:
                break

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
            item_h = 0.2
            if pool_list:
                item_h = float(pool_list[0].get("height", 0.2))
            # Dead center of the floor: farthest from either x-extreme, so it
            # stays clear of the chamfered corner regardless of which side
            # it's actually on (see ContainerState._apply_cut_corner_keepout).
            x = 0.0
            y = 0.0
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
