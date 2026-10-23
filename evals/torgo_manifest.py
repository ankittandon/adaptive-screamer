"""Build a speaker-disjoint train/eval manifest from a TORGO directory.

    python torgo_manifest.py <torgo-dir> <out.jsonl> [--eval F01,M01]

Each line: {"audio": <relpath>, "text": <ref>, "speaker": <spk>, "split": "train"|"eval"}.
Held-out speakers (``--eval``) go to eval so WER measures generalization, not
memorization; everything else is train. Reuses the verbatim-prompt filter from
torgo_wer.py so TORGO's non-speech prompts (image-description tasks, bracketed
instructions, file paths) are excluded. Paths are relative to <torgo-dir> so the
same manifest works once the audio is uploaded to a Modal volume.
"""

import argparse
import glob
import json
import os
import re
from collections import Counter


def valid_ref(t: str):
    t = t.strip()
    low = t.lower()
    if not t:
        return None
    if any(x in t for x in ["[", "]", "/"]) or any(x in low for x in [".jpg", ".bmp", "xxx"]):
        return None
    if not re.search(r"[a-zA-Z]", t):
        return None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("torgo")
    ap.add_argument("out")
    ap.add_argument("--split-mode", choices=["speaker", "utterance"], default="speaker")
    ap.add_argument("--eval", default="F01", help="speaker-mode: comma-separated held-out speakers")
    ap.add_argument("--eval-frac", type=float, default=0.15, help="utterance-mode: eval fraction per speaker")
    args = ap.parse_args()
    eval_spk = {s.strip() for s in args.eval.split(",") if s.strip()}

    rows = []
    for pf in glob.glob(os.path.join(args.torgo, "*/Session*/prompts/*")):
        try:
            ref = valid_ref(open(pf, errors="ignore").read())
        except OSError:
            continue
        if not ref:
            continue
        base = os.path.splitext(os.path.basename(pf))[0]
        sess = os.path.dirname(os.path.dirname(pf))
        spk = pf.split(os.sep)[-4]
        wav = None
        for mic in ("wav_headMic", "wav_arrayMic"):
            cand = os.path.join(sess, mic, base + ".wav")
            if os.path.exists(cand):
                wav = cand
                break
        if not wav:
            continue
        rows.append({
            "audio": os.path.relpath(wav, args.torgo),
            "text": ref,
            "speaker": spk,
        })

    # Assign splits.
    if args.split_mode == "speaker":
        for r in rows:
            r["split"] = "eval" if r["speaker"] in eval_spk else "train"
    else:  # utterance: hold out a deterministic fraction per speaker
        for spk in {r["speaker"] for r in rows}:
            srt = sorted((r for r in rows if r["speaker"] == spk), key=lambda r: r["audio"])
            n_eval = max(1, round(len(srt) * args.eval_frac))
            step = max(1, len(srt) // n_eval)
            for i, r in enumerate(srt):
                r["split"] = "eval" if i % step == 0 else "train"

    rows.sort(key=lambda r: (r["split"], r["speaker"], r["audio"]))
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    c = Counter(r["split"] for r in rows)
    print(f"wrote {len(rows)} utterances -> {args.out} | train={c['train']} eval={c['eval']}")
    for spk in sorted({r["speaker"] for r in rows}):
        sc = Counter(r["split"] for r in rows if r["speaker"] == spk)
        print(f"  {spk}: train={sc['train']} eval={sc['eval']}")
    if c["train"] == 0 or c["eval"] == 0:
        print("WARNING: one split is empty — check --eval speakers against what's present.")


if __name__ == "__main__":
    main()
