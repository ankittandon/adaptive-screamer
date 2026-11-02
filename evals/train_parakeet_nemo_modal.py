"""Fine-tune an NVIDIA Parakeet ASR model on TORGO dysarthric speech with NeMo,
on Modal — the "Best Use of NVIDIA" track for Fireworks: Deploy.

This mirrors train_whisper_modal.py but on the NVIDIA stack: an NVIDIA open
model (Parakeet) + NVIDIA's NeMo framework, run on NVIDIA GPUs. It reuses the
torgo-data volume already populated for the Whisper run (audio + manifest.jsonl).

SMOKE-TEST FIRST (the night before), in this order — each is cheap:
    modal run evals/train_parakeet_nemo_modal.py::build_manifest     # ~1 min, CPU
    modal run evals/train_parakeet_nemo_modal.py::evaluate           # base Parakeet WER (frontier baseline)
Then on the day:
    modal run evals/train_parakeet_nemo_modal.py::finetune           # the training run
    modal run evals/train_parakeet_nemo_modal.py::evaluate --nemo-path /data/parakeet-torgo.nemo

RISK NOTES (read before the day):
- The NeMo NGC container is the robust install path. If the nvcr.io pull needs
  auth on Modal, add an NGC secret, or fall back to the pip install block below.
- transcribe() return type varies by NeMo version (list[str] vs list[Hypothesis]);
  _texts() handles both.
- Parakeet-0.6B fine-tunes fine on one A10G/L4; bump to A100 if OOM. Flag GPU cost.
- For ~900 utterances, freezing the encoder (FREEZE_ENCODER=True) resists overfit
  and trains faster — the recommended first run.
"""

import modal

# NVIDIA open model (CC-BY-4.0). Alternatives: nvidia/parakeet-ctc-0.6b,
# nvidia/canary-1b (multitask). Keep English; TORGO is English read speech.
MODEL = "nvidia/parakeet-tdt-0.6b-v2"
FREEZE_ENCODER = True

app = modal.App("parakeet-torgo")

# Robust path: NVIDIA's own NeMo container already has torch + NeMo + CUDA.
image = (
    modal.Image.from_registry("nvcr.io/nvidia/nemo:24.09", add_python=None)
    .entrypoint([])
    .pip_install("jiwer", "soundfile")
)
# Fallback if the NGC pull is blocked (slower, more fragile):
# image = (modal.Image.debian_slim(python_version="3.10")
#          .apt_install("git", "libsndfile1", "ffmpeg")
#          .pip_install("Cython", "nemo_toolkit[asr]==1.23.0", "jiwer", "soundfile"))

vol = modal.Volume.from_name("torgo-data", create_if_missing=True)

import re

_norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def _texts(out):
    """NeMo transcribe() returns list[str] or list[Hypothesis] across versions."""
    return [o if isinstance(o, str) else getattr(o, "text", str(o)) for o in out]


@app.function(image=image, volumes={"/data": vol}, timeout=60 * 30)
def build_manifest():
    """Convert /data/manifest.jsonl -> NeMo manifests (adds the required duration
    field, keeps the speaker tag for stratified WER)."""
    import json

    import soundfile as sf

    rows = [json.loads(l) for l in open("/data/manifest.jsonl")]
    counts = {}
    for split in ("train", "eval"):
        n = 0
        with open(f"/data/nemo_{split}.json", "w") as out:
            for r in rows:
                if r["split"] != split:
                    continue
                text = r["text"].strip().lower()
                path = f"/data/torgo/{r['audio']}"
                if not text:
                    continue
                try:
                    dur = sf.info(path).duration
                except Exception:
                    continue
                out.write(
                    json.dumps(
                        {
                            "audio_filepath": path,
                            "duration": dur,
                            "text": text,
                            "speaker": r["speaker"],
                        }
                    )
                    + "\n"
                )
                n += 1
        counts[split] = n
    vol.commit()
    print(f"manifests written: {counts}")
    return counts


def _wer_by_speaker(rows, hyps):
    import jiwer

    buckets, allr, allh = {}, [], []
    for r, h in zip(rows, hyps):
        ref, hyp = _norm(r["text"]), _norm(h)
        if not ref:
            continue
        buckets.setdefault(r["speaker"], ([], []))
        buckets[r["speaker"]][0].append(ref)
        buckets[r["speaker"]][1].append(hyp)
        allr.append(ref)
        allh.append(hyp)
    sev = {"F01": "severe", "F03": "moderate", "F04": "mild"}
    print(f"{'stratum':<20}{'n':>5}{'WER':>9}")
    for spk in sorted(buckets):
        rs, hs = buckets[spk]
        print(f"{spk + ' (' + sev.get(spk, '?') + ')':<20}{len(rs):>5}{jiwer.wer(rs, hs) * 100:>8.1f}%")
    overall = jiwer.wer(allr, allh) * 100
    print(f"{'Overall':<20}{len(allr):>5}{overall:>8.1f}%")
    return overall


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60)
def evaluate(nemo_path: str = MODEL, split: str = "eval"):
    """Stratified WER for a Parakeet checkpoint (pretrained id or a .nemo on the
    volume). Run once on the base model for the frontier baseline, again on the
    fine-tuned .nemo."""
    import json

    import nemo.collections.asr as nemo_asr

    if nemo_path.endswith(".nemo"):
        model = nemo_asr.models.ASRModel.restore_from(nemo_path)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(nemo_path)

    rows = [json.loads(l) for l in open(f"/data/nemo_{split}.json")]
    files = [r["audio_filepath"] for r in rows]
    hyps = _texts(model.transcribe(files, batch_size=16))
    print(f"\n=== {nemo_path} on {split} ({len(rows)} utt) ===")
    return _wer_by_speaker(rows, hyps)


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60 * 3)
def finetune(epochs: int = 10, lr: float = 1e-4, batch: int = 8):
    """Fine-tune Parakeet on the TORGO train manifest; save a .nemo to the volume."""
    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    model = nemo_asr.models.ASRModel.from_pretrained(MODEL)

    with open_dict(model.cfg):
        model.cfg.train_ds.manifest_filepath = "/data/nemo_train.json"
        model.cfg.train_ds.batch_size = batch
        model.cfg.train_ds.is_tarred = False
        model.cfg.validation_ds.manifest_filepath = "/data/nemo_eval.json"
        model.cfg.validation_ds.batch_size = batch
        model.cfg.optim.lr = lr
        model.cfg.optim.sched.warmup_steps = 100

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    if FREEZE_ENCODER:
        model.encoder.freeze()  # small-data adaptation: train the decoder/head
        print("encoder frozen")

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=epochs,
        precision="bf16-mixed",
        enable_checkpointing=False,
        logger=False,
        val_check_interval=1.0,
    )
    model.set_trainer(trainer)
    trainer.fit(model)

    out = "/data/parakeet-torgo.nemo"
    model.save_to(out)
    vol.commit()
    print(f"\n=== saved {out} — now: modal run ...::evaluate --nemo-path {out} ===")
    return out


@app.local_entrypoint()
def main():
    """Default run = build manifests then baseline the pretrained model."""
    print(build_manifest.remote())
    print("base Parakeet WER:", evaluate.remote())
