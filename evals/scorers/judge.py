"""Fireworks LLM judges for the two metrics a rules pass can't capture: whether
the styled rewrite preserved the speaker's intent, and whether it is genuinely
good writing.

Both compare the raw transcript (what the speaker said) with the styled text
(what the layer pasted) and return a score in 0..1 plus a one-line reason. The
judge model is called at temperature 0 for stability.

Key resolution mirrors the app: `FIREWORKS_API_KEY`, else the key file at
`~/Library/Application Support/Screamer/fireworks.key`. With no key available the
judges return `valid=False` so a run (e.g. CI) degrades gracefully instead of
failing.

Note: judging real exported dictations sends that text to Fireworks — the same
provider that already processes your dictation in the styling layer. The
committed sample data is synthetic.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_JUDGE_MODEL = os.environ.get(
    "FIREWORKS_JUDGE_MODEL", "accounts/fireworks/models/deepseek-v4p1-flash-1017"
)
_TIMEOUT = 30


@dataclass
class Judgment:
    score: float
    reason: str
    valid: bool


def _key_file() -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "Screamer"
        / "fireworks.key"
    )


def resolve_key() -> Optional[str]:
    env = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if env:
        return env
    try:
        k = _key_file().read_text().strip()
        return k or None
    except OSError:
        return None


def _parse_score(text: str) -> Optional[tuple[float, str]]:
    """Pull {"score","reason"} out of the model's reply, tolerating stray prose
    or code fences around the JSON."""
    # Prefer a JSON object if one is present.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            score = float(obj["score"])
            reason = str(obj.get("reason", "")).strip()
            return max(0.0, min(1.0, score)), reason
        except (ValueError, KeyError, TypeError):
            pass
    # Fallback: a bare number somewhere in the reply.
    num = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if num:
        return max(0.0, min(1.0, float(num.group(1)))), text.strip()[:200]
    return None


def _call(system: str, user: str, model: str) -> Judgment:
    key = resolve_key()
    if not key:
        return Judgment(0.0, "no Fireworks key (set FIREWORKS_API_KEY)", valid=False)
    try:
        resp = requests.post(
            ENDPOINT,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.0,
                "max_tokens": 300,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:  # network, HTTP, shape — all non-fatal for a run
        return Judgment(0.0, f"judge call failed: {e}", valid=False)

    parsed = _parse_score(content)
    if parsed is None:
        return Judgment(0.0, f"could not parse judge reply: {content[:150]}", valid=False)
    score, reason = parsed
    return Judgment(score, reason, valid=True)


INTENT_SYSTEM = """\
You evaluate whether a cleaned-up rewrite of dictated speech preserves the \
speaker's original meaning and intent.

You are given ORIGINAL (raw speech-to-text) and REWRITE (the cleaned version). \
Cleaning up filler, false starts, punctuation, and capitalization is expected \
and good. What you are checking is that the REWRITE did not change what the \
speaker meant.

Give a high score when the rewrite keeps every claim, keeps the speech act (a \
question stays a question, a request stays a request, a maybe stays a maybe), \
and keeps the hedging and tone. Give a low score when the rewrite adds or drops \
information, alters a claim, turns a question into a statement, hardens a hedge \
into a certainty, resolves an ambiguity the speaker left open, or otherwise \
shifts the intent.

Respond with only a JSON object: {"score": <number 0..1>, "reason": "<one short sentence>"}."""

QUALITY_SYSTEM = """\
You evaluate the writing quality of a cleaned-up rewrite of dictated speech.

You are given ORIGINAL (raw speech-to-text) and REWRITE (the cleaned version). \
Judge the REWRITE as a piece of written text: is it clear, concise, naturally \
phrased, and correctly punctuated, and does it read like something the speaker \
would have written if they had typed it, keeping their voice?

Lower the score for writing that sounds like AI: em or en dashes, buzzwords \
(delve, seamless, leverage, robust, tapestry), "not X but Y" constructions, \
rule-of-three lists, rhetorical questions, or signpost phrases. Also lower it \
for text that is clunky, wordy, or awkward.

Respond with only a JSON object: {"score": <number 0..1>, "reason": "<one short sentence>"}."""


def judge_intent(raw: str, styled: str, model: str = DEFAULT_JUDGE_MODEL) -> Judgment:
    if styled.strip() == raw.strip():
        return Judgment(1.0, "unchanged — intent trivially preserved", valid=True)
    user = f"ORIGINAL:\n{raw}\n\nREWRITE:\n{styled}"
    return _call(INTENT_SYSTEM, user, model)


def judge_quality(raw: str, styled: str, model: str = DEFAULT_JUDGE_MODEL) -> Judgment:
    user = f"ORIGINAL:\n{raw}\n\nREWRITE:\n{styled}"
    return _call(QUALITY_SYSTEM, user, model)
