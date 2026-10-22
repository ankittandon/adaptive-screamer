"""Generate styled text from a Fireworks model on a held-out set and print the
averaged slop / intent / quality — the before/after number for the demo.

The Eval Protocol test (test_styling_ab.py) is the dashboard-native version;
this is the terminal report. Same styling system prompt, same scorers.

    python ab_report.py --dataset /tmp/styling_eval.jsonl \
      --model accounts/fireworks/models/qwen3-8b                 # baseline
    python ab_report.py --dataset /tmp/styling_eval.jsonl \
      --model accounts/<acct>/models/screamer-styling-v1         # tuned

Needs FIREWORKS_API_KEY (or ~/Library/Application Support/Screamer/fireworks.key).
"""

import argparse
import json
import os
import sys

import requests

from scorers.judge import judge_intent, judge_quality, resolve_key
from scorers.slop import slop_score
from test_styling_ab import SYSTEM_PROMPT  # reuse the one styling prompt

ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"


def style(key: str, model: str, raw: str) -> str:
    body = {
        "model": model,
        "temperature": 0.3,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw},
        ],
    }
    r = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", help="Fireworks model id to generate from (not the litellm prefix)")
    p.add_argument(
        "--stored",
        action="store_true",
        help="score the already-stored pasted_text (current production styling) instead of generating",
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if not args.stored and not args.model:
        print("pass --model to generate, or --stored to score the stored output", file=sys.stderr)
        return 2

    key = resolve_key()
    if not key:
        print("no Fireworks key", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    label = "stored pasted_text (production)" if args.stored else args.model
    slops, intents, quals = [], [], []
    for i, r in enumerate(rows):
        raw = (r.get("raw_text") or "").strip()
        if not raw:
            continue
        if args.stored:
            styled = (r.get("pasted_text") or "").strip()
            if not styled:
                continue
        else:
            try:
                styled = style(key, args.model, raw)
            except Exception as e:
                print(f"  [{i}] generation failed: {e}", file=sys.stderr)
                continue
        s = slop_score(styled)
        ji = judge_intent(raw, styled)
        jq = judge_quality(raw, styled)
        slops.append(s)
        if ji.valid:
            intents.append(ji.score)
        if jq.valid:
            quals.append(jq.score)

    n = len(slops)
    if n == 0:
        print("no rows scored", file=sys.stderr)
        return 1
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"\nmodel: {label}")
    print(f"scored: {n} held-out inputs")
    print(f"  avg slop/1000:   {avg(slops):.1f}  (lower is better)")
    print(f"  avg intent:      {avg(intents):.3f}  (higher is better, n={len(intents)})")
    print(f"  avg quality:     {avg(quals):.3f}  (higher is better, n={len(quals)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
