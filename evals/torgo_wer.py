"""Measure base-Whisper WER on TORGO dysarthric speech (the accessibility baseline).

    python torgo_wer.py <torgo-dir> [model=base] [limit=0]

Pairs each prompts/<id>.txt with wav_headMic/<id>.wav (falls back to arrayMic),
skips TORGO's non-verbatim prompts (image-description tasks, bracketed
instructions, file paths), transcribes with faster-whisper, and reports WER.
"""

import glob
import os
import re
import sys
import wave

import numpy as np
from faster_whisper import WhisperModel
import jiwer


def read_wav(path: str) -> np.ndarray:
    """TORGO clips are 16 kHz mono 16-bit; read straight to a float32 array so
    faster-whisper never touches PyAV (which is version-skewed here)."""
    w = wave.open(path, "rb")
    raw = w.readframes(w.getnframes())
    w.close()
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

TORGO = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "base"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0


def valid_ref(t: str):
    t = t.strip()
    low = t.lower()
    if not t:
        return None
    # Skip non-verbatim prompts: image-description tasks, instructions, file paths.
    if any(x in t for x in ["[", "]", "/"]) or any(x in low for x in [".jpg", ".bmp", "xxx"]):
        return None
    if not re.search(r"[a-zA-Z]", t):
        return None
    return t


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


pairs = []
for pf in glob.glob(os.path.join(TORGO, "*/Session*/prompts/*")):
    try:
        ref = open(pf, errors="ignore").read()
    except OSError:
        continue
    ref = valid_ref(ref)
    if not ref:
        continue
    base = os.path.splitext(os.path.basename(pf))[0]
    sess = os.path.dirname(os.path.dirname(pf))
    wav = None
    for mic in ("wav_headMic", "wav_arrayMic"):
        cand = os.path.join(sess, mic, base + ".wav")
        if os.path.exists(cand):
            wav = cand
            break
    if wav:
        spk = pf.split(os.sep)[-4]  # F01 / F03 / F04
        pairs.append((wav, ref, spk))

pairs.sort()
if LIMIT:
    pairs = pairs[:LIMIT]
print(f"valid wav+ref pairs: {len(pairs)}")

model = WhisperModel(MODEL, device="cpu", compute_type="int8")
refs, hyps, spks = [], [], []
for i, (wav, ref, spk) in enumerate(pairs):
    segs, _ = model.transcribe(read_wav(wav), language="en", beam_size=1)
    hyp = " ".join(s.text for s in segs)
    refs.append(ref)
    hyps.append(hyp)
    spks.append(spk)
    if i % 100 == 0:
        print(f"  {i}/{len(pairs)}", file=sys.stderr)

rN = [norm(r) for r in refs]
hN = [norm(h) for h in hyps]
keep = [(r, h, s) for r, h, s in zip(rN, hN, spks) if r.strip()]
rF = [r for r, h, s in keep]
hF = [h for r, h, s in keep]

print(f"\n=== sample (ref -> hyp) ===")
for (wav, ref, spk), h in list(zip(pairs, hN))[:4]:
    print(f"  [{spk}] {norm(ref)[:50]!r} -> {h[:50]!r}")

overall = jiwer.wer(rF, hF)
print(f"\nmodel: whisper-{MODEL} (faster-whisper, cpu int8)")
print(f"utterances scored: {len(rF)}")
print(f"BASELINE WER on TORGO dysarthric (F): {overall*100:.1f}%")

# per-speaker
spks_f = [s for _, _, s in keep]
for spk in sorted(set(spks_f)):
    r = [rF[j] for j in range(len(rF)) if spks_f[j] == spk]
    h = [hF[j] for j in range(len(hF)) if spks_f[j] == spk]
    print(f"  {spk}: WER {jiwer.wer(r, h)*100:.1f}%  (n={len(r)})")
