"""Summarize a directory of scripts.run_test result JSON files (produced by
running the configs from gen_benchmark_configs.py) into one readable table.

Usage (from the simulator's root, after running each bench config through
scripts.run_test into the same --result-dir):
    python summarize_bench_results.py --result-dir results/ --prefix bench_
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--prefix", default="bench_", help="only summarize files matching <prefix>*.json")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.result_dir, f"{args.prefix}*.json"))):
        with open(path) as f:
            data = json.load(f)
        for task_name, result in data.items():
            evaluation = result.get("evaluation") or {}
            rows.append({
                "task": task_name,
                "status": result.get("status"),
                "fill_score": evaluation.get("fill_score"),
                "num_placed_items": evaluation.get("num_placed_items"),
                "is_included": (result.get("place_states") or {}).get("is_included"),
                "is_valid": (result.get("place_states") or {}).get("is_valid"),
                "is_placed_safe": (result.get("place_states") or {}).get("is_placed_safe"),
                "message": (result.get("message") or "")[:80],
            })

    name_w = max((len(r["task"]) for r in rows), default=4)
    header = f'{"task":<{name_w}}  {"status":<12}  {"fill":>6}  {"placed":>7}  {"incl":>5}  {"valid":>5}  {"safe":>5}'
    print(header)
    print("-" * len(header))
    for r in rows:
        fill = f'{r["fill_score"]:.2f}' if isinstance(r["fill_score"], (int, float)) else "-"
        placed = f'{r["num_placed_items"]:.2f}' if isinstance(r["num_placed_items"], (int, float)) else "-"
        print(
            f'{r["task"]:<{name_w}}  {str(r["status"]):<12}  {fill:>6}  {placed:>7}  '
            f'{str(r["is_included"]):>5}  {str(r["is_valid"]):>5}  {str(r["is_placed_safe"]):>5}'
        )
        if r["status"] != "success":
            print(f'    {r["message"]}')

    fills = [r["fill_score"] for r in rows if isinstance(r["fill_score"], (int, float))]
    if fills:
        print()
        print(f"mean fill_score across {len(fills)} scenario(s): {sum(fills) / len(fills):.2f}")


if __name__ == "__main__":
    main()
