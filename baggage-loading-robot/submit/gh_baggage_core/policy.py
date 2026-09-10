"""Online (sequential) placement policy.

For every visible pool item x every orientation x every container, we ask
the heightmap packer (packing.best_position, via selection.rank_placements)
for its best landing spot per item, then pick among the top few candidates
using a shallow lookahead: for each, simulate a short greedy continuation
over the *rest of the current pool* (which we can already see -- this isn't
peeking at the future stream) and prefer whichever first move leaves the
best few next steps, not just the locally cheapest one. See selection.py
for how a single step's cost combines stack height, priority/soft placement
rules, and the collision-risk penalties (blocked entry path,
unsupported/tipping perch, forced corner detour) that this lookahead is
meant to catch a step earlier than a purely greedy choice would.
"""
from __future__ import annotations

import time

import numpy as np

from .container_state import ContainerState
from .selection import largest_first_order, pick_with_lookahead, rank_placements

# The evaluation harness enforces an 8-10s wall-clock budget per policy()
# call and, on timeout, substitutes a *random* action of its own -- which is
# far more likely to fail the transport-path/settle checks and end the whole
# episode early than a merely-suboptimal choice of ours would be. We budget
# well under that limit so slower/loaded evaluation hardware can never push
# us over it, and degrade gracefully (return the best candidate found so
# far) rather than risk the harness's own fallback.
TIME_BUDGET_SECONDS = 5.5

# How many of the current pool's best first-moves to actually branch on, and
# how many additional greedy steps to simulate per branch. Kept small: cost
# is roughly BRANCH * STEPS * pool_size * 12 best_position calls on top of
# the base rank_placements pass, and pool_size can be up to ~40, all within
# a single policy() call's 8-10s budget (unlike the offline planner, which
# can afford to look much further ahead -- see offline_planner.py).
LOOKAHEAD_BRANCH = 3
LOOKAHEAD_STEPS = 1


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

        ranked = rank_placements(states, candidates, deadline)
        if not ranked:
            return self._fallback_action(observation)

        if len(ranked) > 1 and time.perf_counter() < deadline:
            chosen = pick_with_lookahead(
                states, candidates, ranked, deadline,
                branch=LOOKAHEAD_BRANCH, steps=LOOKAHEAD_STEPS,
            )
        else:
            chosen = ranked[0]

        _, candidate_idx, c_idx, orn_idx, result, _dims = chosen
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
