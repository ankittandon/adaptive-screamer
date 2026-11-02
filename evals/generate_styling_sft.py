"""Build the anti-slop SFT dataset for fine-tuning Nemotron: fresh raw dictations
(disjoint from the eval set) paired with clean "gold" rewrites produced by a
strong harness model. Writes data/styling_sft.jsonl in Fireworks chat format.

    python generate_styling_sft.py [--n-per 16]

Each line: {"messages":[{system: harness},{user: raw},{assistant: clean}]}.
The targets are distilled from deepseek-flash + our anti-slop harness (fast, and
it follows the rules well). The eval set (data/styling_eval.jsonl) is held out —
we drop any generated dictation whose text collides with it.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from generate_styling_eval import SCENARIOS, SYS as GEN_SYS, _key, gen_one  # reuse
from styling_headtohead import HARNESS_PROMPT, _clean_output

ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
CLEAN_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash-1017"

_norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def clean_one(raw):
    key = _key()
    for _ in range(3):
        try:
            r = requests.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": CLEAN_MODEL, "temperature": 0.2, "max_tokens": 1024,
                      "messages": [{"role": "system", "content": HARNESS_PROMPT},
                                   {"role": "user", "content": raw}]},
                timeout=90,
            )
            r.raise_for_status()
            styled = _clean_output(r.json()["choices"][0]["message"].get("content") or "")
            if styled:
                return styled
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=16)
    a = ap.parse_args()

    eval_texts = {_norm(json.loads(l)["raw"]) for l in open("data/styling_eval.jsonl")}

    jobs = [(i, s) for i, s in enumerate(s2 for s2 in SCENARIOS for _ in range(a.n_per))]
    with ThreadPoolExecutor(max_workers=10) as ex:
        raws = [r for r in ex.map(gen_one, jobs) if r]
    # drop collisions with the eval set
    raws = [r for r in raws if _norm(r["raw"]) not in eval_texts]
    print(f"generated {len(raws)} fresh dictations (eval-disjoint)")

    with ThreadPoolExecutor(max_workers=10) as ex:
        cleaned = list(ex.map(clean_one, [r["raw"] for r in raws]))

    pairs = []
    for r, clean in zip(raws, cleaned):
        if clean and len(clean) > 10:
            pairs.append({"messages": [
                {"role": "system", "content": HARNESS_PROMPT},
                {"role": "user", "content": r["raw"]},
                {"role": "assistant", "content": clean},
            ]})
    dest = Path("data/styling_sft.jsonl")
    dest.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
    print(f"wrote {len(pairs)} SFT pairs -> {dest}")


if __name__ == "__main__":
    main()
