"""Push eval scores back to Braintrust so they sit on the same trace rows.

Braintrust rows are keyed by the telemetry record id, which is also the
`ExportRow` id the eval suite reads. So we compute the eval metrics for each row
and merge a `scores` object into the matching Braintrust row by id
(`_is_merge: true`) — the scores land on the dictation they belong to, alongside
the text/audio the app's sink already shipped.

Scores pushed (all 0..1, higher is better):
- `slop`            — rules-based, always computed (free, offline).
- `intent_fidelity` — LLM judge, only with --judges and a Fireworks key.
- `quality`         — LLM judge, only with --judges and a Fireworks key.

Usage:
    screamer-push-scores --dataset export.jsonl [--judges] [--limit N] [--dry-run]

The dataset is an `ExportRow` JSONL from `screamer-telemetry-export` (the same
format the eval suite reads). Its ids must match the Braintrust rows, so use your
real export, not the synthetic sample, when pushing for real.

Config: Braintrust key from BRAINTRUST_API_KEY or
`~/Library/Application Support/Screamer/braintrust.key`; project from
BRAINTRUST_PROJECT (default below); endpoint from BRAINTRUST_ENDPOINT. The
judges additionally need a Fireworks key (see scorers/judge.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from scorers.judge import judge_intent, judge_quality
from scorers.slop import slop_score, slop_to_score

DEFAULT_PROJECT = os.environ.get(
    "BRAINTRUST_PROJECT", "92bf6a63-16e3-450c-ba02-2f3247e7ef92"
)
DEFAULT_ENDPOINT = os.environ.get("BRAINTRUST_ENDPOINT", "https://api.braintrust.dev/v1")
BATCH = 100


def _braintrust_key() -> Optional[str]:
    env = os.environ.get("BRAINTRUST_API_KEY", "").strip()
    if env:
        return env
    try:
        k = (
            Path.home() / "Library" / "Application Support" / "Screamer" / "braintrust.key"
        ).read_text().strip()
        return k or None
    except OSError:
        return None


def _scores_for(row: Dict[str, Any], with_judges: bool) -> Dict[str, float]:
    raw = row.get("raw_text") or ""
    styled = row.get("pasted_text") or ""
    scores: Dict[str, float] = {"slop": slop_to_score(slop_score(styled))}
    if with_judges:
        ji = judge_intent(raw, styled)
        if ji.valid:
            scores["intent_fidelity"] = ji.score
        jq = judge_quality(raw, styled)
        if jq.valid:
            scores["quality"] = jq.score
    return scores


def _load(dataset: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(dataset) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("id"):
                rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Push eval scores back to Braintrust.")
    p.add_argument(
        "--dataset",
        default=os.environ.get("SCREAMER_EVAL_DATASET")
        or str(Path(__file__).parent / "data" / "sample.jsonl"),
        help="ExportRow JSONL (ids must match Braintrust rows).",
    )
    p.add_argument("--judges", action="store_true", help="also run the LLM judges (needs a Fireworks key)")
    p.add_argument("--limit", type=int, default=None, help="only the first N rows")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--dry-run", action="store_true", help="print events, do not POST")
    args = p.parse_args(argv)

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"dataset not found: {dataset}", file=sys.stderr)
        return 2

    rows = _load(dataset, args.limit)
    if not rows:
        print(f"no rows in {dataset}", file=sys.stderr)
        return 1

    events = [
        {"id": r["id"], "_is_merge": True, "scores": _scores_for(r, args.judges)}
        for r in rows
    ]

    if args.dry_run:
        for e in events:
            print(json.dumps(e))
        print(f"\n[dry-run] {len(events)} events (not sent)", file=sys.stderr)
        return 0

    key = _braintrust_key()
    if not key:
        print(
            "no Braintrust key: set BRAINTRUST_API_KEY or write braintrust.key",
            file=sys.stderr,
        )
        return 2

    url = f"{args.endpoint.rstrip('/')}/project_logs/{args.project}/insert"
    sent = 0
    for i in range(0, len(events), BATCH):
        chunk = events[i : i + BATCH]
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"events": chunk},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"insert failed (HTTP {resp.status_code}): {resp.text[:300]}", file=sys.stderr)
            return 1
        sent += len(chunk)

    metrics = sorted({k for e in events for k in e["scores"]})
    print(f"pushed scores for {sent} rows to project {args.project} (metrics: {', '.join(metrics)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
