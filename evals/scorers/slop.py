"""A faithful Python port of the Rust slop detector.

Mirrors `crates/screamer-styling/src/slop.rs` exactly (same word lists, same
weights, same per-1000-words normalization and rounding) so the eval suite and
the app agree on a score. Keep the two in sync; `test_slop_parity.py` guards the
anchor cases from the Rust unit tests.

Basis: the buzzword list and excess-vocabulary framing come from Kobak et al.
2025 (Science Advances; arXiv:2406.07016) — see docs/styling-guide.md. The list
is meant to stay living, not fixed.
"""

from __future__ import annotations

import math
from typing import Dict

# Buzzwords that read as AI/corporate filler. Matched as whole words.
BUZZWORDS = {
    "delve", "seamless", "seamlessly", "leverage", "leveraging", "utilize",
    "utilizing", "robust", "tapestry", "realm", "landscape", "testament",
    "underscore", "underscores", "showcasing", "navigating", "elevate",
    "unlock", "unlocking", "furthermore", "moreover", "notably",
}

# Signpost phrases.
SIGNPOSTS = [
    "here's the thing", "the thing is", "at the end of the day",
    "needless to say", "it's worth noting", "that said", "when it comes to",
    "in today's world", "in the world of", "let's dive in", "let's unpack",
]

# Contrast connectors whose preceding clause recently contained "not"/"n't".
_NOT_BUT_NEEDLES = [", but ", ", it's ", ", it is "]


def _count_occurrences(haystack: str, needle: str) -> int:
    """Non-overlapping occurrences, matching the Rust helper."""
    if not needle:
        return 0
    count = 0
    start = 0
    while True:
        pos = haystack.find(needle, start)
        if pos < 0:
            break
        count += 1
        start = pos + len(needle)
    return count


def slop_analyze(text: str) -> Dict[str, int]:
    """Count AI-writing habits in `text`. Keys match the Rust SlopBreakdown."""
    lower = text.lower()
    words = len(text.split())

    em_dashes = lower.count("—") + lower.count("–")

    # Split on non-alphanumeric, drop empties (mirrors Rust's char split).
    tokens = []
    cur = []
    for ch in lower:
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    buzzwords = sum(1 for t in tokens if t in BUZZWORDS)

    signposts = sum(_count_occurrences(lower, p) for p in SIGNPOSTS)

    over_joins = _count_occurrences(lower, ", which ") + _count_occurrences(lower, ", so ")

    not_but = 0
    for needle in _NOT_BUT_NEEDLES:
        start = 0
        while True:
            at = lower.find(needle, start)
            if at < 0:
                break
            window = lower[max(0, at - 40):at]
            if "not " in window or "n't" in window:
                not_but += 1
            start = at + len(needle)

    return {
        "words": words,
        "em_dashes": em_dashes,
        "buzzwords": buzzwords,
        "signposts": signposts,
        "over_joins": over_joins,
        "not_but": not_but,
    }


def slop_score(text: str) -> float:
    """Weighted habits per 1,000 words (lower is more human)."""
    b = slop_analyze(text)
    weighted = (
        b["em_dashes"] * 2.0
        + b["buzzwords"] * 1.5
        + b["signposts"] * 2.0
        + b["over_joins"] * 1.0
        + b["not_but"] * 2.0
    )
    if b["words"] == 0:
        return 0.0
    per = weighted / (b["words"] / 1000.0)
    # Match Rust f32::round (half away from zero) to one decimal place.
    return math.floor(per * 10.0 + 0.5) / 10.0


def slop_to_score(slop: float) -> float:
    """Map a per-1,000-words slop value to a 0..1 score where higher is better.

    0 slop -> 1.0, 15 -> 0.5, >= 30 -> 0.0. Human prose is typically well under
    ~15 (see docs/styling-guide.md), so 0.5 is a lenient floor. Shared by the
    eval suite and the Braintrust score push so they agree.
    """
    return max(0.0, 1.0 - slop / 30.0)
