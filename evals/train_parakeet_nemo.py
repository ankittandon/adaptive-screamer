"""Fine-tune / evaluate NVIDIA Parakeet on TORGO with NeMo — standalone, 100%
NVIDIA stack (no Modal). Run inside the NeMo NGC container on an NVIDIA GPU box
(NVIDIA Brev, DGX Cloud, or a Spark once it's on the desk):

    docker run --gpus all -it \
      -v "$PWD":/ws -v /path/to/torgo:/data/torgo \
      -v /path/to/manifest.jsonl:/data/manifest.jsonl \
      nvcr.io/nvidia/nemo:24.09 \
      python /ws/evals/train_parakeet_nemo.py <cmd> [--data-root /data]

Commands (run in this order):
    manifest    build NeMo manifests (adds the required duration field)
    evaluate    stratified WER for a checkpoint (base id, or a saved .nemo)
    finetune    fine-tune Parakeet on the train manifest -> saves a .nemo

The NeMo logic here is identical to train_parakeet_nemo_modal.py; only the runner
differs (plain argparse instead of Modal). Keep that file as the de-risk fallback.

Baseline with ZERO infra tonight: NVIDIA-hosted Parakeet/Canary at
https://build.nvidia.com (API key) — benchmark base WER on TORGO without a GPU.
"""

import argparse
import json
import os
import re

MODEL = "nvidia/parakeet-tdt-0.6b-v2"  # NVIDIA open model, CC-BY-4.0
FREEZE_ENCODER = True  # small-data adaptation: train the decoder/head

_norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def _texts(out):
    """NeMo transcribe() returns list[str] or list[Hypothesis] across versions."""
    return [o if isinstance(o, str) else getattr(o, "text", str(o)) for o in out]


def cmd_manifest(args):
    """Convert <data-root>/manifest.jsonl -> nemo_{train,eval}.json."""
    import soundfile as sf

    rows = [json.loads(l) for l in open(f"{args.data_root}/manifest.jsonl")]
    counts = {}
    for split in ("train", "eval"):
        n = 0
        with open(f"{args.data_root}/nemo_{split}.json", "w") as out:
            for r in rows:
                if r["split"] != split:
                    continue
                text = r["text"].strip().lower()
                path = f"{args.data_root}/torgo/{r['audio']}"
                if not text or not os.path.exists(path):
                    continue
                try:
                    dur = sf.info(path).duration
                except Exception:
                    continue
                out.write(
                    json.dumps(
                        {"audio_filepath": path, "duration": dur, "text": text, "speaker": r["speaker"]}
                    )
                    + "\n"
                )
                n += 1
        counts[split] = n
    print(f"manifests written: {counts}")


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
    print(f"{'Overall':<20}{len(allr):>5}{jiwer.wer(allr, allh) * 100:>8.1f}%")


def cmd_evaluate(args):
    import nemo.collections.asr as nemo_asr

    ckpt = args.nemo_path
    if ckpt.endswith(".nemo"):
        model = nemo_asr.models.ASRModel.restore_from(ckpt)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(ckpt)
    rows = [json.loads(l) for l in open(f"{args.data_root}/nemo_{args.split}.json")]
    hyps = _texts(model.transcribe([r["audio_filepath"] for r in rows], batch_size=16))
    print(f"\n=== {ckpt} on {args.split} ({len(rows)} utt) ===")
    _wer_by_speaker(rows, hyps)


def cmd_finetune(args):
    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    model = nemo_asr.models.ASRModel.from_pretrained(MODEL)
    with open_dict(model.cfg):
        model.cfg.train_ds.manifest_filepath = f"{args.data_root}/nemo_train.json"
        model.cfg.train_ds.batch_size = args.batch
        model.cfg.train_ds.is_tarred = False
        model.cfg.validation_ds.manifest_filepath = f"{args.data_root}/nemo_eval.json"
        model.cfg.validation_ds.batch_size = args.batch
        model.cfg.optim.lr = args.lr
        model.cfg.optim.sched.warmup_steps = 100
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    if FREEZE_ENCODER:
        model.encoder.freeze()
        print("encoder frozen")
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=args.epochs,
        precision="bf16-mixed",
        enable_checkpointing=False,
        logger=False,
    )
    model.set_trainer(trainer)
    trainer.fit(model)
    out = f"{args.data_root}/parakeet-torgo.nemo"
    model.save_to(out)
    print(f"\n=== saved {out} — now: evaluate --nemo-path {out} ===")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["manifest", "evaluate", "finetune"])
    p.add_argument("--data-root", default="/data")
    p.add_argument("--nemo-path", default=MODEL, help="evaluate: base id or a .nemo")
    p.add_argument("--split", default="eval")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    {"manifest": cmd_manifest, "evaluate": cmd_evaluate, "finetune": cmd_finetune}[args.cmd](args)


if __name__ == "__main__":
    main()
