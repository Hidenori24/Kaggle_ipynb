"""Exercises the Agent's public contract end-to-end with a lightweight,
pure-Python stand-in for the simulator's physics/validator layer, so the
full submit/agent.py pipeline can be sanity-checked without pybullet.

This mock is intentionally simple (axis-aligned boxes, drop-straight-down
settling, no physical push/collision path check) -- it is NOT a substitute
for running scripts/run_test.py against the real simulator kit, which
should be done before submitting.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submit")))

from agent import Agent  # noqa: E402
from gh_baggage_core.geometry import oriented_dims  # noqa: E402


def make_item(index, length, width, height, mass=5.0, is_prioritized=False, is_soft=False):
    return {
        "index": index, "length": length, "width": width, "height": height, "mass": mass,
        "is_prioritized": is_prioritized, "is_soft": is_soft,
        "belongs_to": None, "pos": None, "orn": None,
        "lateralFriction": 0.4, "rollingFriction": 0.01, "spinningFriction": 0.01,
        "restitution": 0.2, "angularDamping": 0.8,
    }


class MockContainer:
    def __init__(self, index, length=2.0, width=1.5, height=1.6, thickness=0.04,
                 cut_x=0.4, cut_y=0.4, is_prioritized=False, shelf=False, offset_x=0.0):
        self.index = index
        self.length, self.width, self.height = length, width, height
        self.thickness, self.cut_x, self.cut_y = thickness, cut_x, cut_y
        self.is_prioritized, self.shelf = is_prioritized, shelf
        self.offset_x = offset_x
        self.packed = []

    def info(self):
        return {
            "index": self.index, "length": self.length, "width": self.width, "height": self.height,
            "cut_x": self.cut_x, "cut_y": self.cut_y, "thickness": self.thickness,
            "center": (self.offset_x, 0.0, self.height / 2.0), "shelf": self.shelf,
            "is_prioritized": self.is_prioritized, "packed_items": list(self.packed),
        }

    def place(self, item, local_pos, orn_idx):
        dl, dw, dh = oriented_dims(item["length"], item["width"], item["height"], orn_idx)
        world_pos = (local_pos[0] + self.offset_x, local_pos[1], local_pos[2])
        packed = dict(item)
        packed.update({
            "pos": world_pos, "orn": (0.0, 0.0, 0.0, 1.0), "belongs_to": self.index,
            "length": dl, "width": dw, "height": dh,
        })
        self.packed.append(packed)


class MockEnv:
    """Minimal single/multi-container stand-in exercising only what the
    Agent's optimize()/policy() consume: get_init_states-shaped dict,
    then a policy() observation each step, with straight-down placement."""

    def __init__(self, containers, item_list, lookahead_k):
        self.containers = containers
        self.all_items = list(item_list)
        self.lookahead_k = lookahead_k
        self.pool = []
        self.cursor = 0

    def set_order(self, order):
        by_index = {item["index"]: item for item in self.all_items}
        self.all_items = [by_index[i] for i in order]

    def reset(self):
        self.pool = []
        self.cursor = 0
        for _ in range(self.lookahead_k):
            self._refill()

    def _refill(self):
        if self.cursor < len(self.all_items):
            self.pool.append(self.all_items[self.cursor])
            self.cursor += 1

    def get_init_states(self):
        return {
            "optimize": True,
            "lookahead_k": self.lookahead_k,
            "container_list": [c.info() for c in self.containers],
        }

    def observation(self):
        return {
            "optimize": True,
            "lookahead_k": self.lookahead_k,
            "container_list": [c.info() for c in self.containers],
            "pool_list": list(self.pool),
        }

    def step(self, action):
        item = self.pool.pop(action["item_idx"])
        container = self.containers[action["container_idx"]]
        container.place(item, action["place_pos"], action["orientation"])
        num_space = self.lookahead_k - len(self.pool)
        for _ in range(num_space):
            self._refill()

    def is_empty(self):
        return len(self.pool) == 0


def test_full_mock_episode_places_every_item_without_crashing():
    containers = [MockContainer(index=0)]
    items = [
        make_item(i, length=0.5 + 0.05 * (i % 3), width=0.35, height=0.22,
                  is_soft=(i % 5 == 0), is_prioritized=(i % 7 == 0))
        for i in range(20)
    ]
    env = MockEnv(containers, items, lookahead_k=5)

    agent = Agent(module_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submit")))
    init_states = env.get_init_states()
    agent.get_init_states(init_states)

    order = agent.optimize(list(env.all_items))
    assert sorted(order) == list(range(len(items)))
    env.set_order(order)
    env.reset()

    steps = 0
    while not env.is_empty() and steps < 200:
        obs = env.observation()
        action = agent.policy(obs)
        assert set(action.keys()) == {"item_idx", "container_idx", "place_pos", "orientation"}
        assert 0 <= action["item_idx"] < len(obs["pool_list"])
        assert 0 <= action["container_idx"] < len(containers)
        assert 0 <= action["orientation"] <= 5
        assert isinstance(action["place_pos"], np.ndarray)
        assert action["place_pos"].shape == (3,)
        env.step(action)
        steps += 1

    assert env.is_empty()
    assert sum(len(c.packed) for c in containers) == len(items)


def test_priority_items_route_to_priority_container_when_present():
    containers = [MockContainer(index=0, is_prioritized=True, offset_x=0.0),
                  MockContainer(index=1, is_prioritized=False, offset_x=2.5)]
    items = [make_item(0, 0.5, 0.35, 0.22, is_prioritized=True)]
    env = MockEnv(containers, items, lookahead_k=1)

    agent = Agent(module_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submit")))
    agent.get_init_states(env.get_init_states())
    env.reset()

    action = agent.policy(env.observation())
    assert action["container_idx"] == 0
