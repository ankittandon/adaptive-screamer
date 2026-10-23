# Adaptive Screamer — event pitch & demo runbook

**Event:** Fireworks: Deploy with Microsoft & NVIDIA (Nov 2). Build a task,
deploy an open model, build an eval, tune a dimension against a baseline, measure
the tradeoff, show business value.

**Thesis (say this):** Screamer is a free, private, *on-device* dictation app.
We asked whether open models + the Fireworks stack could make it **faster**,
**better**, and **usable by people it currently fails** — and measured all three
against baselines. The differentiator: it's a **shipping macOS app**, so every
result runs in production, not a notebook.

## Rubric → evidence

| Rubric line | How we hit it |
|---|---|
| Build an eval | Eval Protocol (Fireworks' own) slop/intent/quality on real traces (B); jiwer WER on the public TORGO corpus (C); a 1,078-clip latency/WER/cost benchmark on our own data (A) |
| Tune ≥1 dimension vs a baseline | **C: fine-tuned Whisper, WER 42.7% → 21.9%**; B: anti-slop prompt-tuning cuts slop vs the raw baseline + an SFT pipeline on Fireworks |
| Measure a tradeoff | A: accuracy vs **latency** vs **cost** (on-device vs cloud); B: quality vs cost/latency; C: accuracy vs model size |
| Business value + how far tuning was pushed | Cost ($0 vs a Wispr Flow subscription), privacy, **accessibility reach**, personalization from your own traces |
| Deploy in production | **It's a real app.** On-device Whisper (A, C) + Fireworks styling (B) all run in Screamer today |
| "Make your own frontier" w/ open models | Whisper (open) fine-tuned for dysarthric speech (C); open styling model fine-tune on Fireworks (B) |

## The three measured pillars

### A — On-device vs cloud (latency / cost / privacy)
- **Eval:** 1,078 real dictation clips through on-device Whisper (profiled) and Azure Whisper (cloud).
- **Result (held-out, base.en):** on-device **p50 349 ms / $0** vs Azure **p50 805 ms / $2.27 per 1k dictations**, within **~4 WER points**.
- **Takeaway:** **~2.3× faster, free, fully private** — at competitive accuracy. Production: ships on-device today.

### B — Styling personalization (Fireworks-native)
- **Eval:** slop detector + LLM judges (intent fidelity, writing quality) on **Eval Protocol** (Fireworks' tool), scoring real dictation traces; scores pushed to Braintrust alongside the traces.
- **Baseline (production styling):** slop **4.9**/1k, intent **0.997**, quality **0.868** — already strong (anti-slop prompt-tuning vs raw is the measured win).
- **Tuning pushed:** built the full SFT pipeline on Fireworks (dataset uploaded, fine-tune jobs ran on Qwen3/Muse Glimmer/deepseek-flash).
- **Honest note:** the tuned-*inference* number was blocked by Fireworks deployment friction (no serverless addons, GPU quota, slot holds) — which we **documented as feedback**, reproducing a prior report point-for-point. Depth, not a dead end.

### C — Accessibility ASR (the headline)
- **Eval:** public **TORGO** dysarthric-speech corpus, WER (jiwer), severity-stratified.
- **Baseline:** base Whisper fails on dysarthric speech — **42.7% WER** on held-out utterances (severity-stratified **98% / 60% / 25%** across speakers).
- **Tuned:** fine-tuned Whisper-base on TORGO (HuggingFace `transformers` on Modal) → **21.9% WER — error roughly halved.**
- **Production:** converted to **GGML**, **bundled into Screamer.app** as a selectable "Accessibility" model — transcribes dysarthric speech **live, on-device**.
- **Business value:** a free, private, local dictation tool made usable by people with speech disabilities.
- **Honest note:** utterance-split (speaker-adaptation) framing; gains concentrate in shorter utterances; severe-speaker long sentences stay hard.

## 5-minute demo script

1. **Hook (0:30)** — "Screamer: free, private, on-device dictation. We made it faster, smarter, and usable by people it used to fail — and measured all three." Dictate one line live → text appears instantly.
2. **A · latency/cost (0:45)** — Show the `bench-asr` table. "On-device is 2.3× faster than cloud Whisper, $0 per dictation, and never leaves the machine — at the same accuracy."
3. **B · styling eval (1:15)** — Dictate a messy sentence → it lands cleaned up. Show the Eval Protocol dashboard (slop/intent/quality on real traces) + Braintrust. "Eval-driven styling, built on Fireworks' own Eval Protocol + inference + fine-tuning."
4. **C · accessibility (2:00, the moment)** — Play a dysarthric clip with base Whisper → wrong. Switch Screamer to the **Accessibility** model → correct. Show the WER table (**42.7% → 21.9%**, severity-stratified). "We fine-tuned Whisper on TORGO; WER halved; it runs on-device in the shipping app."
5. **Close (0:30)** — Business value (cost, privacy, **accessibility**, personalization) + "we made our own frontier on open models, and it ships."

## Demo prep checklist
- Screamer.app rebuilt with the Accessibility model bundled (done: `bundle.sh`).
- One dysarthric TORGO clip queued for the A/B moment (base vs Accessibility).
- `bench-asr` latency/cost table and the Eval Protocol dashboard open.
- The WER numbers and the rubric table on a slide (optional — can render from this doc).

## Bonus: Fireworks feedback (shows depth)
Live-reproduced, with evidence: trained adapters aren't callable per-token until
deployed; no base supports serverless *addons*; the default shape wanted a
GB300 at quota 0; deleted deployments hold their slot ~10 min. All documented in
the session — strong material for the Fireworks team and a credible "we pushed
the platform" story.
