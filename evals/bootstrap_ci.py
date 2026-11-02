"""Bootstrap 95% confidence intervals + a paired significance test for a WER
base-vs-tuned comparison, from an eval_dump JSONL ({speaker, ref, base, tuned}).

    python bootstrap_ci.py <eval_dump_*.jsonl> [--n 10000] [--by-speaker]

WER is corpus-level (total word edits / total reference words). We resample
UTTERANCES with replacement (the independent unit), recompute base WER, tuned
WER, and their difference each time, and report percentile CIs. The one-sided
p-value is the bootstrap fraction where tuned did NOT beat base (delta <= 0) —
evidence the improvement is real, not noise. Clears the rubric's "use confidence
intervals or statistical tests" on 100+ held-out examples.
"""

import argparse
import json
import random
import re

import jiwer

_norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def counts(ref, hyp):
    """(edits, ref_len) in words for one utterance, or None if ref empty."""
    ref = _norm(ref)
    if not ref:
        return None
    o = jiwer.process_words(ref, _norm(hyp))
    edits = o.substitutions + o.deletions + o.insertions
    ref_len = o.substitutions + o.deletions + o.hits  # = reference word count
    return edits, ref_len


def wer(idx, base_c, tuned_c):
    be = sum(base_c[i][0] for i in idx)
    te = sum(tuned_c[i][0] for i in idx)
    rl = sum(base_c[i][1] for i in idx)
    return be / rl * 100, te / rl * 100


def report(name, base_c, tuned_c, n_boot, rng):
    m = len(base_c)
    pt_b, pt_t = wer(range(m), base_c, tuned_c)
    deltas, bws, tws = [], [], []
    for _ in range(n_boot):
        idx = [rng.randrange(m) for _ in range(m)]
        bw, tw = wer(idx, base_c, tuned_c)
        bws.append(bw)
        tws.append(tw)
        deltas.append(bw - tw)
    deltas.sort(); bws.sort(); tws.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot)
    p = sum(1 for d in deltas if d <= 0) / n_boot  # one-sided: tuned no better
    print(f"\n{name}  (n={m} utterances, {n_boot} bootstraps)")
    print(f"  base  WER {pt_b:5.1f}%  95% CI [{bws[lo]:.1f}, {bws[hi]:.1f}]")
    print(f"  tuned WER {pt_t:5.1f}%  95% CI [{tws[lo]:.1f}, {tws[hi]:.1f}]")
    print(f"  reduction {pt_b - pt_t:+5.1f} pts  95% CI [{deltas[lo]:+.1f}, {deltas[hi]:+.1f}]  "
          f"p(tuned not better)={p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--by-speaker", action="store_true")
    a = ap.parse_args()
    rng = random.Random(0)

    rows = [json.loads(l) for l in open(a.dump)]
    base_c, tuned_c, spk = [], [], []
    for r in rows:
        cb = counts(r["ref"], r["base"])
        ct = counts(r["ref"], r["tuned"])
        if cb is None:
            continue
        base_c.append(cb)
        tuned_c.append(ct)
        spk.append(r["speaker"])

    report("OVERALL", base_c, tuned_c, a.n, rng)
    if a.by_speaker:
        sev = {"F01": "severe", "F03": "moderate", "F04": "mild"}
        for s in sorted(set(spk)):
            bc = [base_c[i] for i in range(len(spk)) if spk[i] == s]
            tc = [tuned_c[i] for i in range(len(spk)) if spk[i] == s]
            report(f"{s} ({sev.get(s, '?')})", bc, tc, a.n, rng)


if __name__ == "__main__":
    main()
