"""Generate a held-out set of synthetic RAW dictation transcripts for the styling
head-to-head benchmark. Writes data/styling_eval.jsonl (one {id, scenario, raw}
per line). Run once; the set is committed so the benchmark is reproducible.

    python generate_styling_eval.py [--n-per 5]

Uses a small fast Fireworks model (deepseek-flash) only to *author varied inputs*
— the inputs are then styled by every arm of the benchmark, so authoring-model
choice doesn't bias the comparison. Key: FIREWORKS_API_KEY or the app key file.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
GEN_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash-1017"

SCENARIOS = [
    "a work email to a colleague", "a slack message to your team",
    "a meeting note to yourself", "a text to a friend making plans",
    "a code review comment", "a bug report", "a product idea",
    "a reply to a customer", "a journal entry about your day",
    "a reminder about errands", "a status update to your manager",
    "a brainstorm about a feature", "notes from a phone call",
    "a message rescheduling a meeting", "a quick thought while coding",
    "feedback on a design", "a to-do list spoken aloud",
    "an explanation of a technical decision", "a message to a landlord",
    "a recap of a conversation",
]

SYS = (
    "You produce ONE realistic RAW speech-to-text dictation transcript, exactly as "
    "an on-device Whisper model would output it BEFORE any cleanup. Make it messy and "
    "natural: mostly lowercase, little or no punctuation, filler words (um, uh, like, "
    "you know, i mean), at least one false start or self-correction, run-on phrasing, "
    "and occasionally one plausible mis-transcription of a word. 1 to 4 sentences. "
    "Output ONLY the raw transcript text — no quotes, labels, or commentary."
)


def _key():
    k = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if k:
        return k
    return (Path.home() / "Library/Application Support/Screamer/fireworks.key").read_text().strip()


def gen_one(args):
    idx, scenario = args
    key = _key()
    for attempt in range(3):
        try:
            r = requests.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": GEN_MODEL, "temperature": 1.0, "max_tokens": 600,
                    "messages": [
                        {"role": "system", "content": SYS},
                        {"role": "user", "content": f"Scenario: {scenario}."},
                    ],
                },
                timeout=90,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content")
            raw = (content or "").strip().strip('"').strip()
            if len(raw) >= 15:
                return {"id": f"dict-{idx:03d}", "scenario": scenario, "raw": raw}
        except Exception as e:
            if attempt == 2:
                print(f"  gen {idx} failed: {e}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=5)
    a = ap.parse_args()
    jobs = []
    i = 0
    for s in SCENARIOS:
        for _ in range(a.n_per):
            jobs.append((i, s))
            i += 1
    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(gen_one, jobs):
            if res:
                out.append(res)
    dest = Path("data/styling_eval.jsonl")
    dest.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"wrote {len(out)} dictations -> {dest}")


if __name__ == "__main__":
    main()
