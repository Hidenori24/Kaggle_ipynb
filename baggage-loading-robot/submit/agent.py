"""SIGNATE Baggage-Loading Robot challenge: submission entry point.

Implements the required `Agent` interface (`__init__`, `get_init_states`,
`optimize`, `policy`) described in the official simulator README. The
actual packing logic lives in `gh_baggage_core/` (imported via
`module_path`, which the evaluation harness always passes in as the
directory this file lives in).

Design summary (see ../docs/DESIGN.md for the full write-up):

- Offline `optimize`: when offline optimization is enabled, the online
  phase only ever sees one item at a time (lookahead_k == 1), so this is
  the only place any real sequencing decision can be made. Rather than a
  static sort, it runs a full dry-run simulation of the same search the
  online policy uses (gh_baggage_core/offline_planner.py), greedily
  choosing the best-fitting remaining item at each step. Falls back to a
  simple largest-first/soft-last heuristic (gh_baggage_core/ordering.py)
  if the planner can't run at all (e.g. no container info was cached).
- Online `policy`: rebuilds an exact top-down occupancy heightmap per
  container from the ground-truth `packed_items` positions given every
  step (not the rendered depth image, which would only be an approximation
  of the same information), then searches every (pool item, orientation,
  container) combination for the lowest, flattest landing spot, subject to:
    * prioritized bags never go into a non-prioritized container when a
      prioritized one exists,
    * non-prioritized/non-soft items avoid landing on top of a
      prioritized/soft item whenever an alternative spot exists.
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_importable(module_path: str | None) -> None:
    for path in (module_path, _THIS_DIR):
        if path and path not in sys.path:
            sys.path.insert(0, path)


class Agent:
    def __init__(self, module_path: str):
        _ensure_importable(module_path)
        from gh_baggage_core.offline_planner import plan_order
        from gh_baggage_core.ordering import offline_order
        from gh_baggage_core.policy import Policy

        self._plan_order = plan_order
        self._offline_order = offline_order
        self._policy_cls = Policy
        self._policy = None
        self._container_list = None

    def get_init_states(self, init_states: dict) -> bool:
        lookahead_k = init_states.get("lookahead_k") if isinstance(init_states, dict) else None
        self._container_list = init_states.get("container_list") if isinstance(init_states, dict) else None
        self._policy = self._policy_cls(lookahead_k=lookahead_k)
        return True

    def optimize(self, item_list: list) -> list[int]:
        try:
            if self._container_list:
                order = self._plan_order(self._container_list, item_list)
                if order is not None and set(order) == {item["index"] for item in item_list}:
                    return order
        except Exception:
            pass
        try:
            return self._offline_order(item_list)
        except Exception:
            # Never let a bug here abort the whole evaluation: fall back to
            # the identity order, which is always a valid permutation.
            return [item["index"] for item in item_list]

    def policy(self, observation: dict) -> dict:
        if self._policy is None:
            self._policy = self._policy_cls(lookahead_k=observation.get("lookahead_k"))
        return self._policy.act(observation)
