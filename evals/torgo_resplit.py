"""Re-split an existing TORGO manifest to remove overfitting shortcuts, so the
eval measures real generalization rather than memorized vocabulary/voices.

    python torgo_resplit.py <in.jsonl> <out.jsonl> --mode text    [--eval-frac 0.15]
    python torgo_resplit.py <in.jsonl> <out.jsonl> --mode speaker --eval F01

Modes:
  text     eval holds utterances whose (normalized) text does NOT appear in
           train — kills closed-vocabulary memorization (TORGO repeats prompts).
  speaker  eval holds whole speakers (unseen voice). Note: speakers read the same
           prompts, so this alone still shares vocabulary — pair with `text`.

Prints a leakage check: for `text` it must report 0% text overlap.
"""

import argparse
import json
import re
from collections import Counter

_norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["text", "speaker"], required=True)
    ap.add_argument("--eval", default="F01", help="speaker mode: comma-separated held-out speakers")
    ap.add_argument("--eval-frac", type=float, default=0.15, help="text mode: fraction of unique texts held out")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]

    if args.mode == "speaker":
        eval_spk = {s.strip() for s in args.eval.split(",") if s.strip()}
        for r in rows:
            r["split"] = "eval" if r["speaker"] in eval_spk else "train"
    else:  # text-disjoint: partition unique texts deterministically
        texts = sorted({_norm(r["text"]) for r in rows if _norm(r["text"])})
        n_eval = max(1, round(len(texts) * args.eval_frac))
        step = max(1, len(texts) // n_eval)
        eval_texts = {t for i, t in enumerate(texts) if i % step == 0}
        for r in rows:
            r["split"] = "eval" if _norm(r["text"]) in eval_texts else "train"

    rows.sort(key=lambda r: (r["split"], r["speaker"], r["audio"]))
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    c = Counter(r["split"] for r in rows)
    tr_txt = {_norm(r["text"]) for r in rows if r["split"] == "train"}
    ev_txt = [_norm(r["text"]) for r in rows if r["split"] == "eval"]
    leak = sum(1 for t in ev_txt if t in tr_txt)
    print(f"{args.mode}: wrote {len(rows)} -> {args.out} | train={c['train']} eval={c['eval']}")
    print(f"  text overlap eval∩train: {leak}/{len(ev_txt)} = {leak / max(1, len(ev_txt)) * 100:.0f}%")
    for spk in sorted({r["speaker"] for r in rows}):
        sc = Counter(r["split"] for r in rows if r["speaker"] == spk)
        print(f"  {spk}: train={sc['train']} eval={sc['eval']}")
    if c["train"] == 0 or c["eval"] == 0:
        print("WARNING: a split is empty.")


if __name__ == "__main__":
    main()
