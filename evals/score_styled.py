"""Score pre-generated styled outputs (slop + intent + quality) into head-to-head
result rows. For the NVIDIA fine-tuned styling model, whose inference runs on
Modal — this scores the returned text with the same metrics as the live runner.

    python score_styled.py <styled.jsonl> <arm_label> <out.jsonl> [--workers 8]

<styled.jsonl> rows: {id, raw, styled}. Judges run on Fireworks (inference is not
blocked — only dataset uploads are).
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "scorers")
from slop import slop_score  # noqa: E402
from judge import INTENT_SYSTEM, QUALITY_SYSTEM  # noqa: E402
from styling_headtohead import _judge  # noqa: E402


def score_row(args):
    row, arm = args
    styled = (row.get("styled") or "").strip()
    if not styled:
        return {"id": row["id"], "arm": arm, "raw": row["raw"], "error": "empty"}
    return {
        "id": row["id"], "arm": arm, "raw": row["raw"], "styled": styled,
        "slop": slop_score(styled),
        "intent": _judge(INTENT_SYSTEM, row["raw"], styled),
        "quality": _judge(QUALITY_SYSTEM, row["raw"], styled),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("styled")
    ap.add_argument("arm")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.styled)]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(score_row, [(r, a.arm) for r in rows]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in results) + "\n")
    import statistics as st
    ok = [r for r in results if "slop" in r]
    print(f"scored {len(ok)}/{len(results)} -> {a.out}")
    if ok:
        print(f"  mean slop={st.mean(r['slop'] for r in ok):.1f} "
              f"intent={st.mean(r['intent'] for r in ok if r['intent'] is not None):.3f} "
              f"quality={st.mean(r['quality'] for r in ok if r['quality'] is not None):.3f}")


if __name__ == "__main__":
    main()
