"""Offline optimization: decide the order items enter the conveyor stream.

The online `policy` still gets final say over which pool item to place next
and where (it can look ahead across `lookahead_k` items), so this ordering
mainly shapes *which items are visible together* early on. We apply two
well-established bin-packing heuristics:

- Large-items-first (by volume): a stronger geometric foundation is laid
  down first, which tends to reduce fragmentation for what follows.
- Soft items pushed later: combined with the online policy's refusal to
  place non-soft items on top of soft ones, streaming soft items later
  makes it more likely they land near the top of a stack (as intended)
  rather than needing to be dug around.
"""
from __future__ import annotations


def offline_order(item_list: list[dict]) -> list[int]:
    def sort_key(item: dict):
        volume = item.get("volume", item["length"] * item["width"] * item["height"])
        is_soft = bool(item.get("is_soft", False))
        is_prioritized = bool(item.get("is_prioritized", False))
        # (soft last, priority items nudged slightly earlier within their
        # bucket, largest volume first)
        return (is_soft, not is_prioritized, -volume)

    ordered = sorted(item_list, key=sort_key)
    return [item["index"] for item in ordered]
