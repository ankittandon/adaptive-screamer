# Adaptive Screamer — Fireworks: Deploy submission

**Task:** make on-device dictation **adaptable to each user** — at the *ear* (how
you speak) and the *pen* (how you write).
**One-line result:** at both layers, a specialized open model / harness **beats a
frontier baseline** on held-out data — dysarthric ASR with WER roughly halved
(statistically significant), and a styling harness that strips AI-slop while
preserving intent — all in a shipping, on-device macOS app.

- **Judging commit:** latest on `main` (run `git rev-parse HEAD`); see end of file
- **Repository:** https://github.com/ankittandon/adaptive-screamer (public)
- **Sponsor tracks entered:** **Fireworks** (styling + Eval Protocol), **NVIDIA Best-Use** (on-device ASR)

---

## 1. Project overview

**The idea: one dictation app that adapts to you at two layers.** Screamer is a
real, free, private, on-device macOS dictation app. We make it **adaptable** — it
learns *how you speak* and *how you write* — using specialized open models and
harnesses that beat frontier baselines, measured honestly.

- **The ear — ASR adaptation (NVIDIA / on-device).** Fine-tune the open ASR model
  to a speaker's voice, including **dysarthric speech** that general ASR fails. On
  held-out data, WER is roughly halved with statistically significant gains;
  converted to GGML and shipped on-device as a selectable model.
- **The pen — styling adaptation (Fireworks).** A harness that rewrites raw
  transcripts into clean written text in the user's *own* voice — fixing filler,
  punctuation, and mis-transcriptions while **removing AI-slop** and preserving
  intent. Runs on a Fireworks open model and is evaluated on Fireworks' **Eval
  Protocol** (slop + intent + quality judges).
- **The flywheel.** Both layers improve **per-user, on-device, during downtime** —
  continuous adaptation that keeps the model yours and your data local.

**Who benefits.** Everyone who dictates — it's free, private, and local — and
especially **people with dysarthria** (cerebral palsy, ALS, stroke), who need
hands-light input most and whom general ASR serves worst.

**Why it matters.** A dictation tool that adapts to the individual: usable by
people it used to fail, cleaner output in your own voice, at $0 and fully private.

Supporting result — **efficiency:** on-device ASR is ~2.3× faster than cloud at
$0 and no cold start (§3).

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

**McNemar's exact test** (paired, on whole-utterance exact-match rate, tuned vs
base) corroborates the bootstrap: utterance p=1.4×10⁻¹³, speaker-disjoint
(unseen voice) p=0.021. On text-disjoint the *WER* reduction is significant
(bootstrap, above) while the stricter *whole-utterance-correct* rate rises +5.7 pts
at p=0.08 — expected, since McNemar is all-or-nothing and WER gives partial credit.
(`evals/mcnemar.py`.)

**How we know it isn't overfitting.** See
[docs/case-study-accessibility.md](docs/case-study-accessibility.md#how-we-know-it-isnt-overfitting):
the win survives removing both the vocabulary and the speaker shortcut; checkpoints
are selected by held-out WER; and the continuous-learning loop's validation gate
refuses to promote a model that doesn't generalize.

**Reproduce it.** `evals/torgo_resplit.py` (splits), `evals/train_whisper_modal.py`
(`train` / `eval_strata` / `eval_dump`), `evals/bootstrap_ci.py` (CIs),
`evals/nvidia_asr_bench.py` (NVIDIA-hosted baseline). See `evals/README.md`.

---

## 2b. The pen — styling head-to-head (Fireworks)

Same rigor for the styling layer, **100% on Fireworks** (styling models + a
*neutral* qwen-max judge — neither contestant). **120 held-out synthetic
dictations**, three arms:
- **A1** frontier (`kimi-k3`) + naive "clean this up" prompt — realistic baseline
- **A2** frontier (`kimi-k3`) + our anti-slop harness — fair strong baseline
- **A3** ours — `deepseek-flash` (small, what Screamer ships) + our harness

| Arm | slop/1k | intent | quality | latency | tokens |
|---|--:|--:|--:|--:|--:|
| A1 frontier + naive | 27.5 | 0.919 | 0.872 | 5587 ms | 450 |
| A2 frontier + our harness | 3.5 | 0.983 | 0.949 | 8502 ms | 970 |
| **A3 ours (flash + harness)** | **8.4** | **0.979** | **0.927** | **2669 ms** | 802 |

Paired **bootstrap 95% CIs** + **McNemar exact test** (on slop-free rate):

- **Ours vs frontier out-of-box (A3 vs A1):** we win on *every* axis — slop
  −17.4/1k [10.7, 23.8] p<0.001, intent +0.05 [0.02, 0.09] p=0.001, quality +0.05
  [0.02, 0.08] p<0.001; slop-free rate 73.7% vs 46.6% (**+27 pts, McNemar
  p=1.4×10⁻⁵**), at ~2× lower latency.
- **Ours vs frontier under the same harness (A3 vs A2) — the disclosed tradeoff:**
  intent **tied** (CI includes 0), quality within noise (−0.02), and the frontier
  model is somewhat cleaner (slop-free 89% vs 74%, McNemar p=0.004) — but **ours is
  ~3.2× faster** (2.7 s vs 8.5 s) and uses fewer tokens.

**Claim:** the anti-slop **harness is the lever** — it cuts slop ~8× on *any*
model. Our small open model **beats a frontier model's out-of-box styling on slop,
intent, and quality (all significant)**, and **matches a frontier model running the
same harness on intent/quality at ~3× lower latency/cost**, trading a little
slop-cleanliness for the efficiency (gains retain useful quality, per rule 8).

Reproduce: `evals/generate_styling_eval.py`, `evals/styling_headtohead.py`,
`evals/styling_ci.py`, `evals/mcnemar.py`; results in
`evals/results/styling_headtohead.jsonl`.

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

---

**Judging commit:** submit the current `main` HEAD at submission time
(`git rev-parse HEAD`). This document and the code are versioned together.
