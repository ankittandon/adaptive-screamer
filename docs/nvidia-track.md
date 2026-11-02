# "Best Use of NVIDIA" track — day-of runbook (Nov 2)

**Prize:** an NVIDIA **DGX Spark** for the Best Use of NVIDIA category.
**Event:** 10:30 build start, **4:30 submit sharp**, finals 7:00. 6 hours.

## The core idea — dual track, don't replace Whisper

Our crown jewel is **on-device**: Screamer runs the fine-tuned **Whisper** model
locally (GGML/whisper.cpp/Metal), dysarthric WER **42.7% → 22.3%**. That stays
the live demo, untouched and guaranteed.

The NVIDIA category is won **alongside** it, not by swapping it out:

- **NVIDIA frontier push:** fine-tune NVIDIA's open **Parakeet** on TORGO in
  **NeMo**, on **NVIDIA GPUs** — "an open NVIDIA model pushes the dysarthric-speech
  frontier," ideally beating our own Whisper number. Deploy/serve via **NIM**.
- **Why not just run Parakeet in Screamer?** Parakeet is a NeMo
  Conformer/TDT model — **whisper.cpp can't run it**, so it does not drop into
  Screamer's pipeline. On-device Parakeet needs a different runtime
  (**sherpa-onnx / ONNX Runtime**). That's the stretch goal + the funded next
  phase, not the core demo.
- **The Spark pitch writes itself:** on-device Parakeet-in-Screamer = private
  on-prem fine-tuning + a local ONNX runtime = exactly what a DGX Spark unlocks.

## Fallback ladder (ceiling → floor — we stay in the category at every rung)

- **A (win it):** fine-tuned Parakeet (NeMo) beats base Parakeet *and* our Whisper
  on TORGO, served via NIM; stretch: it runs locally via sherpa-onnx on the Mac.
- **B:** fine-tuned Parakeet beats base Parakeet (stratified), served via NIM.
- **C:** *benchmark* base Parakeet/Canary (NVIDIA open models) vs our Whisper on
  TORGO + deploy one via NIM. Still "used NVIDIA open model + NVIDIA deploy."
- **D (floor):** everything trained on NVIDIA GPUs (name it) + the
  DGX-Spark-enables-the-next-phase pitch. Our Whisper on-device result stands on
  its own.

Even D keeps us in the category. A/B win it. **Decision gates below.**

## The stack — 100% NVIDIA (Modal is only the fallback)

| Layer | NVIDIA-native | Fallback |
|---|---|---|
| Compute (train) | **NVIDIA Brev** or **DGX Cloud** | Modal (still NVIDIA GPUs) |
| Framework | **NeMo** | — |
| Containers + models | **NGC** (NeMo container, Parakeet) | — |
| Inference / deploy | **NIM** — self-host, or **build.nvidia.com** (hosted) | — |
| Optimize (optional) | **TensorRT** | — |

No local NVIDIA GPU on the Mac, so training runs on NVIDIA's cloud until a Spark
is on the desk. Two scripts, same NeMo logic:
- [`evals/train_parakeet_nemo.py`](../evals/train_parakeet_nemo.py) — standalone,
  runs in the NeMo NGC container on Brev/DGX Cloud (the NVIDIA-native path).
- [`evals/train_parakeet_nemo_modal.py`](../evals/train_parakeet_nemo_modal.py) —
  Modal wrapper of the same logic (de-risk fallback; the torgo-data volume
  already has the audio + manifest from the Whisper run).

## Pre-stage TONIGHT (so NeMo isn't a cold start)

1. **No-infra NVIDIA baseline first:** get an API key at **build.nvidia.com** and
   benchmark NVIDIA-hosted **Parakeet/Canary** on a handful of TORGO eval clips.
   Zero GPU provisioning, instantly "used NVIDIA" — this alone secures rung C.
2. **Stand up NVIDIA compute:** create an **NVIDIA Brev** account (or DGX Cloud if
   you have credits) and launch a single-GPU instance (A10G/L4/A100) with the
   **`nvcr.io/nvidia/nemo:24.09`** container. Mount TORGO + manifest.jsonl.
3. **Build manifests + baseline base Parakeet** on that box:
   ```bash
   python evals/train_parakeet_nemo.py manifest   --data-root /data
   python evals/train_parakeet_nemo.py evaluate   --data-root /data
   ```
   Record the stratified base Parakeet WER — the number the fine-tune must beat.
4. **Fallback ready:** if Brev/DGX provisioning stalls tonight, run the same two
   steps on Modal (`modal run evals/train_parakeet_nemo_modal.py::build_manifest`
   then `::evaluate`) so you have a baseline either way.
5. **(Optional) sherpa-onnx recon:** confirm a pre-exported Parakeet ONNX runs
   locally via `sherpa-onnx` on the Mac on one clip — if it works tonight, the
   on-device stretch (NVIDIA model, on the laptop) becomes realistic tomorrow.
6. **At the event:** ask mentors about **DGX Cloud / Brev credits or Spark access**
   — using NVIDIA's own compute live is the strongest possible category signal.

## Day-of timeline (10:30 → 4:30)

| Time | Do | Gate |
|---|---|---|
| 10:30–11:00 | Team + env; pull repo; Modal auth; re-confirm base Parakeet baseline from last night. | |
| 11:00–12:30 | Launch `finetune` (encoder frozen first run). Meanwhile wire the NIM serving path for base Parakeet. | **12:30: is training progressing?** No → drop to rung C (benchmark + NIM). |
| 12:30–1:30 | Lunch; training runs. | |
| 1:30–2:30 | `evaluate --nemo-path /data/parakeet-torgo.nemo` → stratified tuned WER. Compare vs base Parakeet + our Whisper (42.7→22.3). | **2:30: did it beat base?** No → present rung C honestly. |
| 2:30–3:15 | Deploy leg: serve via **NIM**. Stretch: export fine-tuned Parakeet → ONNX → run locally via **sherpa-onnx** on the Mac (on-device NVIDIA model). | **3:15: stretch working?** No → drop it, keep NIM. |
| 3:15–4:00 | Fold numbers into the deck; add the NVIDIA slide + the Spark pitch. Rehearse. | |
| 4:00–4:30 | **Submit (hard buffer).** | |

## Deploy leg — NIM (the "Deploy" in the event name)

Serve base and/or fine-tuned Parakeet through an **NVIDIA NIM** microservice and
show a live transcription through it. This is the deployment half of the story and
reinforces the sponsor's own product. If NIM setup is slow, time-box it — the
fine-tune result is the headline; NIM is the deployment proof point.

## Judge pitch — Best Use of NVIDIA + why we get the Spark

1. **NVIDIA open model:** we fine-tuned **Parakeet** (not just used it) to push the
   **dysarthric-speech frontier** — open model beating the baseline, the event's
   whole thesis, on the NVIDIA stack.
2. **End-to-end NVIDIA stack:** trained in **NeMo** on **NVIDIA's own cloud**
   (Brev / DGX Cloud), models + containers from **NGC**, deployed via **NIM** — no
   third-party compute in the loop.
3. **Why the Spark, specifically:** our next phase is **private, on-prem
   fine-tuning on real consented dysarthric-user data** + **per-user on-device
   adapters** — sensitive data never leaves the box, and inference already runs
   locally on the user's Mac. A DGX Spark is the one piece of kit that makes that
   privacy-preserving loop real. We are the team that would use a Spark for
   exactly what it's for.

## Guaranteed fallback

If the entire NVIDIA track stalls, the submission is still strong: the on-device
Whisper accessibility result (42.7% → 22.3%, stratified, shipping in Screamer),
Pillar A efficiency, and the eval harness — all unchanged. The NVIDIA track is
upside, never a dependency.
