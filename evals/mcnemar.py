"""McNemar's test (exact binomial) + paired bootstrap for two paired systems.

Two modes:
  asr      per-utterance binary = exact normalized match; compares base vs tuned.
           python mcnemar.py asr <eval_dump.jsonl>        (fields: ref, base, tuned)
  styling  per-dictation binary = slop-free (slop == 0); compares two arms.
           python mcnemar.py styling <results.jsonl> --a A3_ours --b A1_frontier_naive

McNemar tests the discordant pairs: b = first-wrong/second... we frame it as
  n01 = base(or arm B) right, tuned(or arm A) wrong
  n10 = base(or arm B) wrong, tuned(or arm A) right
The exact two-sided binomial p-value tests H0: n01 == n10 (no net improvement).
Bootstrap resamples items for a 95% CI on the accuracy (or slop-free rate) gain.
"""

import argparse
import json
import math
import random
import re

_norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def exact_binom_two_sided(n10, n01):
    """Two-sided exact McNemar p-value: P(X deviates as far from n/2), X~Binom(n,0.5)."""
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def bootstrap_rate_gain(pairs, n_boot, rng):
    """pairs = list of (a_success, b_success) bools; gain = rate(a) - rate(b)."""
    m = len(pairs)
    boot = []
    for _ in range(n_boot):
        sa = sb = 0
        for _ in range(m):
            a, b = pairs[rng.randrange(m)]
            sa += a
            sb += b
        boot.append((sa - sb) / m * 100)
    boot.sort()
    return boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)]


def report(name, pairs, n_boot, rng):
    # pairs = (a_success, b_success); a = the system we claim is better
    ra = sum(a for a, _ in pairs) / len(pairs) * 100
    rb = sum(b for _, b in pairs) / len(pairs) * 100
    n10 = sum(1 for a, b in pairs if a and not b)   # a right, b wrong
    n01 = sum(1 for a, b in pairs if b and not a)   # b right, a wrong
    p = exact_binom_two_sided(n10, n01)
    lo, hi = bootstrap_rate_gain(pairs, n_boot, rng)
    print(f"\n{name}  (n={len(pairs)})")
    print(f"  success rate: A={ra:.1f}%  B={rb:.1f}%  gain={ra - rb:+.1f} pts  95% CI [{lo:+.1f}, {hi:+.1f}]")
    print(f"  McNemar discordant: A-only={n10}  B-only={n01}  exact two-sided p={p:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["asr", "styling"])
    ap.add_argument("path")
    ap.add_argument("--a", default="A3_ours")
    ap.add_argument("--b", default="A1_frontier_naive")
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()
    rng = random.Random(0)

    if args.mode == "asr":
        rows = [json.loads(l) for l in open(args.path)]
        pairs = []
        for r in rows:
            ref = _norm(r["ref"])
            if not ref:
                continue
            tuned_ok = _norm(r["tuned"]) == ref
            base_ok = _norm(r["base"]) == ref
            pairs.append((tuned_ok, base_ok))  # A = tuned, B = base
        report(f"ASR exact-match: tuned (A) vs base (B) — {args.path.split('/')[-1]}", pairs, args.n, rng)
    else:
        by = {}
        for l in open(args.path):
            x = json.loads(l)
            if "slop" in x and x.get("slop") is not None:
                by.setdefault(x["arm"], {})[x["id"]] = x["slop"]
        ids = set(by.get(args.a, {})) & set(by.get(args.b, {}))
        pairs = [(by[args.a][i] == 0.0, by[args.b][i] == 0.0) for i in ids]  # slop-free?
        report(f"styling slop-free rate: {args.a} (A) vs {args.b} (B)", pairs, args.n, rng)


if __name__ == "__main__":
    main()
