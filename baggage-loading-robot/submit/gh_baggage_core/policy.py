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
from .geometry import NUM_ORIENTATIONS, oriented_dims
from .packing import best_effort_position
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

        # Any placement our own checks flag as unstable, path-blocked or in
        # the corner keepout maps onto a validator check that ends the whole
        # episode when it fails (env.step terminates on check_transport_path
        # or place_item returning False) -- so taking one costs every item
        # still in the stream, not just this one.
        #
        # Measured on the real simulator: in all four scenarios traced, the
        # placement that ended the run was already flagged here before we
        # made it. only_000 went out on an unstable pick (support 0.625,
        # flat 0.528) that toppled 0.78m; only_001 on a path-blocked pick
        # made with nine other candidates available; shelf on a pick with
        # support 0.125 and core support 0.000. We were not blind to any of
        # them, we just had nothing scoring better and placed them anyway.
        #
        # So ask for a genuinely safe placement first, across every item in
        # the pool. Giving up this item's own spot for a worse-but-safe one
        # (or, at the extreme, a corner of bare floor that scores nothing at
        # all) is near-always worth it against losing the rest of the
        # stream. Only when nothing in the pool has a safe placement
        # anywhere do we fall back to the old least-bad answer.
        ranked = rank_placements(states, candidates, deadline, require_safe=True)
        if not ranked:
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
        # pool item/orientation/container (an effectively full set of
        # containers for this item's height budget -- see
        # best_effort_position's docstring) or the main search raised.
        # Always place pool item 0: with nothing ranked, there's no
        # search-backed reason to prefer any other pool index, and the
        # item chosen here only has to be *some* legal index (see
        # PlacementValidator.check_action).
        #
        # This used to warp straight to a flat, unverified (0, 0) guess.
        # Reproduced against the real simulator, that guess landed squarely
        # on top of four already-packed items at once (a collision distance
        # of -5cm to -8.6cm, not a near miss) -- turning a spot where nothing
        # was going to fit anyway into the *worst* way to fail it. Searching
        # every container/orientation for the lowest real landing spot
        # (still verified against every already-placed item's exact box)
        # can only do as well or better: a resulting inclusion/ceiling
        # failure ends the episode exactly like any other failure already
        # does, but a resulting collision failure is no longer near-certain.
        container_list = observation.get("container_list") or []
        pool_list = observation.get("pool_list") or []
        if not container_list or not pool_list:
            return {
                "item_idx": 0,
                "container_idx": 0,
                "place_pos": np.array([0.0, 0.0, 0.5], dtype=np.float32),
                "orientation": 0,
            }

        item = pool_list[0]
        length = float(item.get("length", 0.2))
        width = float(item.get("width", 0.2))
        height = float(item.get("height", 0.2))

        best = None  # (z, container_idx, orn_idx, result)
        for c_idx, container in enumerate(container_list):
            try:
                state = ContainerState(container)
            except Exception:
                continue
            for orn_idx in range(NUM_ORIENTATIONS):
                dl, dw, dh = oriented_dims(length, width, height, orn_idx)
                try:
                    result = best_effort_position(state, dl, dw, dh)
                except Exception:
                    continue
                if best is None or result["z"] < best[0]:
                    best = (result["z"], c_idx, orn_idx, result)

        if best is not None:
            _, c_idx, orn_idx, result = best
            place_pos = np.array([result["x"], result["y"], result["z"]], dtype=np.float32)
            return {
                "item_idx": 0,
                "container_idx": c_idx,
                "place_pos": place_pos,
                "orientation": orn_idx,
            }

        # Every container/orientation raised (a malformed observation, not
        # just a full container -- best_effort_position itself always
        # returns something for any real container): the same bare guess
        # as before, strictly as a last resort.
        try:
            state = ContainerState(container_list[0])
            z = min(state.floor_z + height / 2.0, state.ceiling_z - height / 2.0)
            place_pos = np.array([0.0, 0.0, z], dtype=np.float32)
        except Exception:
            place_pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        return {
            "item_idx": 0,
            "container_idx": 0,
            "place_pos": place_pos,
            "orientation": 0,
        }
