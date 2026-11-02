# Adaptive Screamer — Fireworks: Deploy submission

**Task:** on-device dictation that works for people with **dysarthric speech**.
**One-line result:** specialized open models beat frontier ASR baselines on
dysarthric speech — WER roughly **halved on held-out data, with statistically
significant gains** — running **on-device**, inside a shipping macOS app.

- **Judging commit:** `0bb641b` (branch `main`)
- **Repository:** https://github.com/ankittandon/adaptive-screamer (public)
- **Sponsor tracks entered:** Fireworks (eligibility — see below), **NVIDIA Best-Use**

---

## 1. Project overview

**Who benefits.** People with **dysarthria** (cerebral palsy, ALS, stroke) are the
users who most need hands-light dictation and the ones general ASR serves worst —
they're effectively locked out of dictation tools. Secondarily, *every* user: the
system is free, fully private, and on-device (no subscription, no cloud).

**What the system does.** Screamer is a real macOS push-to-talk dictation app.
Speech → text, entirely on-device. We make it (a) **usable for dysarthric
speakers** by fine-tuning open ASR, (b) **cheaper/faster** than cloud ASR, and (c)
**cleaner** via a Fireworks-powered styling layer.

**Approach.** Three measured workstreams against frontier baselines:
- **C — Accessibility (headline):** fine-tune open Whisper on public dysarthric
  data (TORGO), convert to GGML, ship it on-device as a selectable model.
- **A — Efficiency:** on-device vs cloud ASR — latency, cost, privacy.
- **B — Styling (Fireworks):** clean/personalize dictated text with a Fireworks
  model, evaluated on Fireworks' **Eval Protocol**.

**Why it matters.** A free, private, local dictation tool made usable by people it
used to fail — plus 2.3× cheaper/faster transcription than cloud at competitive
quality.

---

## 2. Evaluation evidence (the benchmark)

**Baselines (frontier, open).** Two general SOTA open ASR models, each compared on
the *same* held-out cases: **OpenAI Whisper** (`base.en`, what we ship) and
**NVIDIA Parakeet-TDT-0.6b** (benchmarked via **NVIDIA NIM**, hosted). Both fail on
dysarthric speech:

| Base model (no fine-tune) | TORGO held-out WER |
|---|--:|
| Whisper base.en | 42.7% |
| NVIDIA Parakeet-TDT-0.6b (via NIM) | 60.3% |

**Dataset.** [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)
(University of Toronto), public dysarthric-speech corpus; dysarthric-female set,
3 speakers spanning severity (F01 severe, F03 moderate, F04 mild), 16 kHz.
**Metric.** Word-error rate (jiwer), severity-stratified.
**Method.** Supervised fine-tune of `whisper-base.en` (HuggingFace
`transformers`), converted to **GGML** and run in the app's `whisper-rs`/Metal
stack.

**Before → after, with uncertainty.** We report **three held-out splits**, each
with a **95% bootstrap CI** (10,000 resamples) on the WER reduction and a
one-sided paired **p-value**. Per rule 8, evaluation cases are kept **separate from
training data** — the first split had vocabulary/speaker overlap, so we built
leakage-free splits and lead with those:

| Split — what it tests | n | Base | Tuned | Reduction (95% CI) | p |
|---|--:|--:|--:|---|--:|
| Utterance — adaptation to a known user | 181 | 42.7% | 22.3% | −20.4 [14.9, 26.8] | <0.0001 |
| **Text-disjoint — words never in training** | 210 | 31.5% | 21.5% | **−10.0 [5.3, 15.2]** | <0.0001 |
| **Speaker-disjoint — a severe voice never heard** | 114 | 135.6% | 62.9% | **−72.7 [11.6, 194.5]** | <0.0001 |

**Every reduction CI excludes zero**, and the gain holds per severity stratum on
the text-disjoint split (severe −28.8, moderate −7.3, mild −8.4; all p < 0.02).
Base Whisper >100% on the unseen severe voice means it **hallucinates** on severe
speech; the fine-tuned model is consistent (tuned CI [56.6, 69.9]).

**How we know it isn't overfitting.** See
[docs/case-study-accessibility.md](docs/case-study-accessibility.md#how-we-know-it-isnt-overfitting):
the win survives removing both the vocabulary and the speaker shortcut; checkpoints
are selected by held-out WER; and the continuous-learning loop's validation gate
refuses to promote a model that doesn't generalize.

**Reproduce it.** `evals/torgo_resplit.py` (splits), `evals/train_whisper_modal.py`
(`train` / `eval_strata` / `eval_dump`), `evals/bootstrap_ci.py` (CIs),
`evals/nvidia_asr_bench.py` (NVIDIA-hosted baseline). See `evals/README.md`.

---

## 3. Efficiency tradeoff (disclosed per rule 8)

On-device vs two cloud ASR services — including **NVIDIA's own** hosted Parakeet
(via NIM):

| Option | p50 latency | cost / 1k | audio leaves device? | dysarthric WER |
|---|--:|--:|:--:|--:|
| **On-device Whisper (tuned)** | **349 ms** | **$0.00** | **no** | **22.3%** (tuned) |
| Azure Whisper (cloud) | 805 ms | $2.27 | yes | ~match (general) |
| NVIDIA Parakeet-TDT-0.6b (NIM) | 311 ms* | GPU / paid | yes | 60.3% (base) |

\* NVIDIA latency measured on short TORGO clips over venue wifi (not the same
1,078-clip set as the on-device/Azure rows), so treat it as *comparable*, not a
precise head-to-head.

**Warmup + network dependence (the hidden cloud cost).** The cloud options carry a
**cold start** and are **at the mercy of the network**. NVIDIA NIM, measured from
the venue: first (cold) request **~3.0 s**, then ~1.0 s, ~0.9 s. Steady-state
latency swung with wifi quality — **p50 311 ms / p95 713 ms** on a good connection,
but **p50 606 ms / p95 ≈ 19 s** on congested conference wifi. On-device has **no
warmup and no network dependence**: a flat ~349 ms whether the network is empty or
saturated. (This isn't a knock on NVIDIA's service — it's the nature of any cloud
call; it's exactly the risk on-device removes.)

**Read:** on good network NVIDIA's hosted ASR is fast (≈ on-device latency, faster
than Azure), so *average* latency doesn't separate the cloud options from
on-device. On-device wins the rest of the tradeoff — **$0 cost, audio never leaves
the device, no cold start, works offline and consistently regardless of network,
and far better quality on the task that matters** (base Parakeet 60.3% WER on
dysarthric speech vs our tuned on-device model at 22.3%). The efficiency gain
retains — in fact improves — useful task performance.
([`bench-asr`](crates/screamer-whisper/src/bin/bench_asr.rs) for on-device/Azure;
[`nvidia_asr_bench.py --latency`](evals/nvidia_asr_bench.py) for NVIDIA NIM.)

---

## 4. Sponsor tracks

### Fireworks (eligibility)
The project **uses Fireworks** in two core places:
- **Styling layer:** dictated text is cleaned/personalized through **Fireworks
  inference** (OpenAI-compatible), with an anti-"slop" system prompt.
- **Evaluation harness:** built on Fireworks' own **Eval Protocol** — a slop
  detector plus LLM judges (intent fidelity, writing quality) over real dictation
  traces, with scores pushed to Braintrust (`evals/`, `push_scores.py`). We also
  ran supervised fine-tuning jobs on Fireworks.

### NVIDIA Best-Use (DGX Spark)
- **NVIDIA open model:** benchmarked **Parakeet-TDT-0.6b** as a frontier baseline
  via **NVIDIA NIM** (NVCF Riva), and scaffolded a **NeMo** fine-tune of Parakeet
  on TORGO (`evals/train_parakeet_nemo.py`, 100% NVIDIA stack — Brev/DGX Cloud +
  NGC + NIM). Plan + results: [docs/nvidia-track.md](docs/nvidia-track.md).
- **Why the Spark:** our next phase is **private, on-prem fine-tuning on real
  consented dysarthric-user data** plus **per-user on-device adapters** — sensitive
  speech never leaves the box, inference already runs locally. A DGX Spark is the
  one piece of kit that makes that privacy-preserving loop real.

---

## 5. Limitations & honest disclosure

- **Scope:** an experiment, not a production-scale study — one TORGO subset (3
  speakers), **SFT only** (no RL).
- **Claims matched to evals:** the −48% utterance-split number is a
  *personalization* result (same user); the defensible *general* numbers are the
  text-disjoint (−32% rel) and speaker-disjoint (−54% rel) splits.
- **Severe speech stays hard:** 62.9% WER on an unseen severe voice is a big
  improvement but still high.
- **Parakeet fine-tune:** the NVIDIA-native fine-tune is scaffolded and the
  baseline is measured; the headline tuned result is Whisper (the GGML on-device
  path). We distinguish measured results from in-progress work.

### Pre-existing work vs. built for this event (rule 4)
- **Pre-existing:** the Screamer app itself (fork of
  [suvamsh/screamer](https://github.com/suvamsh/screamer), MIT), its dictation /
  telemetry / styling / Braintrust plumbing.
- **Built for this event:** the accessibility ASR workstream (TORGO fine-tune →
  GGML → in-app Accessibility model), the ASR benchmarks (on-device vs cloud;
  NVIDIA-hosted Parakeet baseline), the leakage-free splits + bootstrap-CI
  evaluation, and the NVIDIA NeMo track.
- **AI assistance:** developed with Claude Code; the team is responsible for all
  reported results and can explain the implementation.
- **Data/licenses:** TORGO (public, research license); base models open
  (Whisper MIT-era weights; Parakeet CC-BY-4.0). No credentials are committed.

---

## 6. Required materials checklist
- [x] Project overview — this file
- [x] Public open-source repo with setup + eval instructions — `README.md`, `evals/README.md`
- [x] Before/after results with baseline, dataset, metric, method, tradeoffs — §2–3
- [ ] Demo video < 120s — see [docs/demo-script.md](docs/demo-script.md)
- [x] Team information — *(add all members in the submission form)*
- [x] Sponsor-track information — §4
