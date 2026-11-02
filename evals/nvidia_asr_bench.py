"""Benchmark NVIDIA-hosted ASR (build.nvidia.com / NVCF Riva) on TORGO —
100% NVIDIA stack, zero GPU infra. Also the teacher relabeler for the
continuous-training loop (point --model at whisper-large-v3 or canary-1b).

    python nvidia_asr_bench.py --torgo /tmp/torgo --split eval \
        --model parakeet-tdt-0.6b-v2 [--limit N]

Key is read from $NVIDIA_API_KEY, else from
~/Library/Application Support/Screamer/nvidia.key (never printed).
"""

import argparse
import json
import os
import re
import sys

import jiwer
import riva.client

# NVCF function IDs (from the account's function listing).
FUNCTION_IDS = {
    "parakeet-tdt-0.6b-v2": "d3fe9151-442b-4204-a70d-5fcc597fd610",
    "parakeet-ctc-1.1b": "1598d209-5e27-4d3c-8079-4751568b1081",
    "canary-1b": "b0e8b4a5-217c-40b7-9b96-17d84e666317",
    "whisper-large-v3": "b702f636-f60c-4a3d-a6f4-f3568c13bd7d",
}
SERVER = "grpc.nvcf.nvidia.com:443"

_norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def _key():
    k = os.environ.get("NVIDIA_API_KEY")
    if k:
        return k.strip()
    p = os.path.expanduser("~/Library/Application Support/Screamer/nvidia.key")
    return open(p).read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torgo", default="/tmp/torgo")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--model", default="parakeet-tdt-0.6b-v2", choices=list(FUNCTION_IDS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--latency", type=int, default=0,
                    help="measure per-request latency on this many clips (skips WER)")
    a = ap.parse_args()

    auth = riva.client.Auth(
        None, True, SERVER,
        [["function-id", FUNCTION_IDS[a.model]], ["authorization", "Bearer " + _key()]],
    )
    asr = riva.client.ASRService(auth)

    rows = [json.loads(l) for l in open(f"{a.torgo}/manifest.jsonl")]
    rows = [r for r in rows if r["split"] == a.split]

    if a.latency:
        import statistics as st
        import time

        WARMUP = 3
        allt = []
        for r in rows[: a.latency]:
            wav = f"{a.torgo}/{r['audio']}"
            cfg = riva.client.RecognitionConfig(
                language_code="en-US", max_alternatives=1, enable_automatic_punctuation=False
            )
            riva.client.add_audio_file_specs_to_config(cfg, wav)
            data = open(wav, "rb").read()
            t = time.perf_counter()
            asr.offline_recognize(data, cfg)
            allt.append((time.perf_counter() - t) * 1000)
        warm = allt[:WARMUP]
        steady = sorted(allt[WARMUP:])
        q = lambda p: steady[int(p * len(steady))]
        print(f"NVIDIA NIM {a.model} latency:")
        print(f"  warmup (first {len(warm)} cold requests): "
              + ", ".join(f"{x:.0f}" for x in warm) + " ms")
        print(f"  steady-state (n={len(steady)}): "
              f"p50={q(0.5):.0f} ms  p95={q(0.95):.0f} ms  mean={st.mean(steady):.0f} ms  min={steady[0]:.0f} ms")
        return

    if a.limit:
        rows = rows[: a.limit]

    buckets, allr, allh = {}, [], []
    for i, r in enumerate(rows):
        wav = f"{a.torgo}/{r['audio']}"
        cfg = riva.client.RecognitionConfig(
            language_code="en-US", max_alternatives=1, enable_automatic_punctuation=False
        )
        riva.client.add_audio_file_specs_to_config(cfg, wav)
        with open(wav, "rb") as f:
            data = f.read()
        try:
            resp = asr.offline_recognize(data, cfg)
            hyp = " ".join(
                res.alternatives[0].transcript for res in resp.results if res.alternatives
            )
        except Exception as e:
            hyp = ""
            print(f"err {wav}: {e}", file=sys.stderr)
        ref, h = _norm(r["text"]), _norm(hyp)
        if ref:
            buckets.setdefault(r["speaker"], ([], []))
            buckets[r["speaker"]][0].append(ref)
            buckets[r["speaker"]][1].append(h)
            allr.append(ref)
            allh.append(h)
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}", file=sys.stderr)

    sev = {"F01": "severe", "F03": "moderate", "F04": "mild"}
    print(f"\n=== NVIDIA-hosted {a.model} on TORGO {a.split} ({len(allr)} utt) ===")
    print(f"{'stratum':<20}{'n':>5}{'WER':>9}")
    for spk in sorted(buckets):
        rs, hs = buckets[spk]
        print(f"{spk + ' (' + sev.get(spk, '?') + ')':<20}{len(rs):>5}{jiwer.wer(rs, hs) * 100:>8.1f}%")
    print(f"{'Overall':<20}{len(allr):>5}{jiwer.wer(allr, allh) * 100:>8.1f}%")


if __name__ == "__main__":
    main()
