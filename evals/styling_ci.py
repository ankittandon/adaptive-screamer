"""Summarize the styling head-to-head: per-arm means (slop/intent/quality +
latency/tokens) and paired bootstrap 95% CIs on the deltas between arms.

    python styling_ci.py [results/styling_headtohead.jsonl] [--n 10000]

Deltas are paired over dictation ids (same inputs), so the comparison is tight.
slop: lower is better (reduction = baseline - arm). intent/quality: higher better.
"""

import argparse
import json
import random
import statistics as st


def load(path):
    by_arm = {}
    for l in open(path):
        r = json.loads(l)
        if "styled" not in r:  # errored row
            continue
        by_arm.setdefault(r["arm"], {})[r["id"]] = r
    return by_arm


def mean_ci(vals, n_boot, rng):
    m = len(vals)
    boot = sorted(sum(vals[rng.randrange(m)] for _ in range(m)) / m for _ in range(n_boot))
    lo, hi = boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)]
    return st.mean(vals), lo, hi


def paired(a, b, metric, by_arm, n_boot, rng, higher_better):
    """delta per id; positive = `a` better than `b`."""
    ids = set(by_arm[a]) & set(by_arm[b])
    deltas = []
    for i in ids:
        va, vb = by_arm[a][i].get(metric), by_arm[b][i].get(metric)
        if va is None or vb is None:
            continue
        deltas.append((va - vb) if higher_better else (vb - va))
    m = len(deltas)
    boot = sorted(sum(deltas[rng.randrange(m)] for _ in range(m)) / m for _ in range(n_boot))
    lo, hi = boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)]
    p = sum(1 for d in boot if d <= 0) / n_boot
    return st.mean(deltas), lo, hi, p, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/styling_headtohead.jsonl")
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    rng = random.Random(0)
    by_arm = load(a.path)

    names = {
        "A1_frontier_naive": "A1 frontier + naive prompt",
        "A2_frontier_harness": "A2 frontier + our harness",
        "A3_ours": "A3 ours (flash + harness)",
    }
    print("=== per-arm means ===")
    print(f"{'arm':<30}{'n':>4}{'slop/1k':>9}{'intent':>8}{'quality':>9}{'latency':>9}{'tokens':>8}")
    for arm in ["A1_frontier_naive", "A2_frontier_harness", "A3_ours"]:
        rows = list(by_arm.get(arm, {}).values())
        if not rows:
            continue
        slop = [r["slop"] for r in rows if r.get("slop") is not None]
        intent = [r["intent"] for r in rows if r.get("intent") is not None]
        qual = [r["quality"] for r in rows if r.get("quality") is not None]
        lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        tok = [r["tokens"] for r in rows if r.get("tokens")]
        print(f"{names[arm]:<30}{len(rows):>4}{st.mean(slop):>9.1f}{st.mean(intent):>8.3f}"
              f"{st.mean(qual):>9.3f}{round(st.mean(lat)):>8}ms{round(st.mean(tok)):>8}")

    print("\n=== paired deltas (95% CI; positive = first arm better) ===")
    comparisons = [
        ("A3_ours", "A1_frontier_naive"),
        ("A3_ours", "A2_frontier_harness"),
        ("A2_frontier_harness", "A1_frontier_naive"),
    ]
    for x, y in comparisons:
        print(f"\n{names[x]}  vs  {names[y]}")
        for metric, higher in [("slop", False), ("intent", True), ("quality", True)]:
            d, lo, hi, p, m = paired(x, y, metric, by_arm, a.n, rng, higher)
            tag = "slop reduction" if metric == "slop" else f"{metric} gain"
            print(f"  {tag:<16} {d:+6.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  p(not better)={p:.3f}  n={m}")


if __name__ == "__main__":
    main()
