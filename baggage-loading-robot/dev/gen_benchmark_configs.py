"""Generate a small, varied battery of synthetic evaluation configs for
local regression testing against the real simulator.

Why this exists: our own dev-kit-free unit tests (tests/) can't catch
everything -- and neither, it turns out, can testing against just the two
sample tasks in configs/sample_config.json. A change (packing.py's
graduated instability penalty) that looked neutral-to-positive on both
sample tasks went on to visibly regress a real SIGNATE submission, almost
certainly because SIGNATE's hidden test cases span container/item
combinations our two local samples don't (see docs/DESIGN.md). This script
generates a deliberately varied set of *additional* local cases -- multiple
containers, a prioritized container, shelf-equipped containers, priority
and soft items, small/large item-size mixes -- that can be run with the
official simulator's `scripts.run_test` before trusting a change enough to
submit it for real.

This script only emits config JSON (the schema documented in the official
simulator README); it does not import or depend on the simulator package
itself, so it can run without the simulator kit present. Running the
configs it produces does require the official kit (see ../README.md).

Usage:
    python gen_benchmark_configs.py --out-dir /path/to/simulator/configs/bench

Then, from the simulator's root:
    python -m scripts.run_test --module-path agents/submit/ \\
        --config-path configs/bench/<name>.json --result-dir results/

Every config here uses a fixed seed, so the generated files are
reproducible and can be regenerated identically on any machine.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random

BASE_ACTION = {
    "keys": {"item_idx": "int", "container_idx": "int", "place_pos": "float", "orientation": "int"},
    "pos_lim": {"low": -100, "high": 100},
    "orientations": [0, 1, 2, 3, 4, 5],
}
BASE_VALIDATOR = {
    "inclusion_margin": -0.005,
    "start_z": 0.08,
    "safety_margin": 0.015,
    "ceiling_margin": 0.018,
    "displacement_threshold": 0.3,
    "angle_displacement_threshold": 45,
    "settle_wait_step": 300,
}
BASE_CAMERA = {
    "num_containers": 1, "target_pos": [0, 0, 0], "distance": 3.0,
    "yaw": 0, "pitch": 0, "roll": 0, "img_width": 64, "img_height": 64,
    "fov": 60, "near_val": 0.1, "far_val": 10.0,
}


def _make_container(rng: random.Random, index: int, is_prioritized: bool, shelf: bool) -> dict:
    length = round(rng.uniform(1.6, 2.4), 2)
    width = round(rng.uniform(1.2, 1.6), 2)
    height = round(rng.uniform(1.4, 1.7), 2)
    return {
        "index": index, "length": length, "width": width, "height": height,
        "thickness": round(rng.uniform(0.03, 0.05), 3), "buffer": 0.0,
        "cut_x": round(length * rng.uniform(0.18, 0.26), 3),
        "cut_y": round(height * rng.uniform(0.2, 0.3), 3),
        "packed_items": [], "require_shelf": shelf, "is_prioritized": is_prioritized,
    }


def _make_item(rng: random.Random, index: int, is_prioritized: bool, is_soft: bool, size: str) -> dict:
    scale = {"small": (0.18, 0.35), "medium": (0.3, 0.6), "large": (0.5, 0.9)}[size]
    item = {
        "index": index,
        "length": round(rng.uniform(*scale), 3),
        "width": round(rng.uniform(*scale), 3),
        "height": round(rng.uniform(0.15, 0.5), 3),
        "mass": round(rng.uniform(1.0, 20.0), 2),
        "is_prioritized": is_prioritized,
        "is_soft": is_soft,
        "lateralFriction": round(rng.uniform(0.3, 0.7), 2),
        "rollingFriction": 0.01,
        "spinningFriction": 0.01,
        "restitution": round(rng.uniform(0.0, 0.3), 2),
    }
    if is_soft:
        item.update(contactStiffness=3000, contactDamping=800, linearDamping=0.8)
    return item


def _make_task(rng: random.Random, *, num_containers: int, num_items: int, size_mix: dict,
                prioritized_container: bool, shelf_containers: bool,
                prioritized_item_frac: float, soft_item_frac: float,
                optimize: bool, look_ahead: int) -> dict:
    containers = [
        _make_container(rng, i, is_prioritized=(prioritized_container and i == 0), shelf=shelf_containers)
        for i in range(num_containers)
    ]
    items = []
    sizes = list(size_mix.keys())
    weights = list(size_mix.values())
    for i in range(num_items):
        size = rng.choices(sizes, weights=weights)[0]
        is_prioritized = rng.random() < prioritized_item_frac
        is_soft = rng.random() < soft_item_frac
        items.append(_make_item(rng, i, is_prioritized, is_soft, size))

    camera = copy.deepcopy(BASE_CAMERA)
    camera["num_containers"] = num_containers  # sizes the env's shared-memory depth-map buffer

    return {
        "containers": {"spacing": 2.5, "container_list": containers},
        "item_stream": {"item_list": items, "look_ahead": look_ahead, "max_space": 1, "visible_pool": []},
        "camera": camera,
        "validator": copy.deepcopy(BASE_VALIDATOR),
        "action": copy.deepcopy(BASE_ACTION),
        "agent": {
            "optimize": optimize,
            "init_timeout": 10.0, "optimization_timeout": 180.0, "policy_timeout": 10.0,
            "allowed_methods": ["get_init_states", "optimize", "policy"],
        },
        "visualizer": {"vis": False, "camera": {"yaw": 0, "pitch": -20}},
    }


# (name, seed, kwargs) -- deliberately varied along every axis the real
# hidden test set is likely to vary (see the simulator README's caution
# that container count/size and item count/order differ per evaluation).
SCENARIOS = [
    ("single_container_offline_small_items", 101, dict(
        num_containers=1, num_items=35, size_mix={"small": 3, "medium": 1},
        prioritized_container=False, shelf_containers=False,
        prioritized_item_frac=0.1, soft_item_frac=0.2, optimize=True, look_ahead=1,
    )),
    ("single_container_online_large_items", 102, dict(
        num_containers=1, num_items=20, size_mix={"medium": 1, "large": 2},
        prioritized_container=False, shelf_containers=False,
        prioritized_item_frac=0.0, soft_item_frac=0.15, optimize=False, look_ahead=8,
    )),
    ("shelf_container_mixed_items", 103, dict(
        num_containers=1, num_items=30, size_mix={"small": 1, "medium": 2, "large": 1},
        prioritized_container=False, shelf_containers=True,
        prioritized_item_frac=0.15, soft_item_frac=0.25, optimize=True, look_ahead=1,
    )),
    ("two_containers_one_prioritized", 104, dict(
        num_containers=2, num_items=40, size_mix={"small": 2, "medium": 2, "large": 1},
        prioritized_container=True, shelf_containers=False,
        prioritized_item_frac=0.25, soft_item_frac=0.2, optimize=True, look_ahead=1,
    )),
    ("two_containers_online_heavy_soft", 105, dict(
        num_containers=2, num_items=45, size_mix={"small": 1, "medium": 1},
        prioritized_container=False, shelf_containers=False,
        prioritized_item_frac=0.1, soft_item_frac=0.4, optimize=False, look_ahead=6,
    )),
    ("dense_small_container_stress", 106, dict(
        num_containers=1, num_items=60, size_mix={"small": 5, "medium": 1},
        prioritized_container=False, shelf_containers=False,
        prioritized_item_frac=0.05, soft_item_frac=0.1, optimize=True, look_ahead=1,
    )),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, help="directory to write <name>.json config files into")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for name, seed, kwargs in SCENARIOS:
        rng = random.Random(seed)
        task = _make_task(rng, **kwargs)
        path = os.path.join(args.out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump({name: task}, f, indent=2)
        print(f"wrote {path} ({len(task['item_stream']['item_list'])} items, "
              f"{len(task['containers']['container_list'])} container(s))")


if __name__ == "__main__":
    main()
