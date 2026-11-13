"""Path segment similarity scoring for closest-match route detection."""

from __future__ import annotations

import re


def _normalize_and_split(path: str) -> list[str]:
    """Normalize path parameters to {param} and split into segments."""
    normalized = re.sub(r"\{[^}]+\}", "{param}", path.rstrip("/"))
    return [s for s in normalized.split("/") if s]


def segment_overlap_score(path_a: str, path_b: str) -> float:
    """Score 0.0-1.0 measuring segment-level similarity between two paths.

    Scoring:
    1. Common segments (order-independent): shared / max_segments
    2. Terminal segment bonus: +0.3 if last segments match
    3. Prefix bonus: +0.2 if one path is a prefix of the other
    4. Length penalty: -0.1 * abs(length_difference)

    Clamped to [0.0, 1.0].
    """
    segs_a = _normalize_and_split(path_a)
    segs_b = _normalize_and_split(path_b)

    if not segs_a or not segs_b:
        return 0.0

    max_len = max(len(segs_a), len(segs_b))

    # Common segments (multiset intersection)
    common = 0
    remaining = list(segs_b)
    for seg in segs_a:
        if seg in remaining:
            common += 1
            remaining.remove(seg)
    base_score = common / max_len

    # Terminal segment bonus
    terminal_bonus = 0.3 if segs_a[-1] == segs_b[-1] else 0.0

    # Prefix bonus
    prefix_bonus = 0.0
    shorter, longer = (segs_a, segs_b) if len(segs_a) <= len(segs_b) else (segs_b, segs_a)
    if longer[: len(shorter)] == shorter:
        prefix_bonus = 0.2

    # Length penalty
    length_penalty = 0.1 * abs(len(segs_a) - len(segs_b))

    return max(0.0, min(1.0, base_score + terminal_bonus + prefix_bonus - length_penalty))
