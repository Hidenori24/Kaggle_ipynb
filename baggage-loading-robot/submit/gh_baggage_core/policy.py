"""Online (sequential) placement policy.

For every visible pool item x every orientation x every container, we ask
the heightmap packer (packing.best_position, via selection.choose_placement)
for its best landing spot, then pick the single (item, orientation,
container, position) combination with the lowest overall cost. See
selection.py for how that cost combines stack height, priority/soft
placement rules, and the collision-risk penalties (blocked entry path,
unsupported/tipping perch, forced corner detour).
"""
from __future__ import annotations

import time

import numpy as np

from .container_state import ContainerState
from .selection import choose_placement, largest_first_order

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

        # Evaluate the biggest items first: a good decision matters most for
        # them, so if the time budget forces an early cutoff we still end up
        # having considered the placements that matter most.
        item_order = largest_first_order(pool_list)
        candidates = [pool_list[i] for i in item_order]

        best = choose_placement(states, candidates, deadline)
        if best is None:
            return self._fallback_action(observation)

        _, candidate_idx, c_idx, orn_idx, result, _dims = best
        item_pos_idx = item_order[candidate_idx]

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
        # Only reached when best_position found nowhere valid for *any*
        # pool item/orientation (an effectively full container) or the main
        # search raised, so there's no guarantee this specific spot is
        # collision-free either -- but it must at least satisfy the
        # inclusion check on its own, which a bare `thickness` floor height
        # does not (see ContainerState.floor_z's SAFETY_MARGIN).
        container_list = observation.get("container_list") or []
        pool_list = observation.get("pool_list") or []
        if container_list:
            try:
                state = ContainerState(container_list[0])
                item_h = float(pool_list[0]["height"]) if pool_list else 0.2
                z = min(state.floor_z + item_h / 2.0, state.ceiling_z - item_h / 2.0)
                place_pos = np.array([0.0, 0.0, z], dtype=np.float32)
            except Exception:
                place_pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        else:
            place_pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        return {
            "item_idx": 0,
            "container_idx": 0,
            "place_pos": place_pos,
            "orientation": 0,
        }
