# Teaching an on-device dictation app to hear dysarthric speech

*How we fine-tuned Whisper on public dysarthric-speech data, cut word-error rate
roughly in half, and shipped it running on-device — built as an experiment on
open models and the Fireworks / Modal stack.*

## TL;DR

- **Challenge.** Screamer is a free, private, on-device dictation app. Its
  Whisper model works well for typical speech but **fails on dysarthric speech**
  (speakers with cerebral palsy, ALS, and similar conditions), effectively
  locking those users out.
- **Approach.** Supervised fine-tuning of **openai/whisper-base.en** on the
  public **TORGO** dysarthric-speech corpus, run on a managed GPU, then converted
  to GGML and loaded into the shipping app.
- **Result.** Word-error rate on held-out dysarthric utterances dropped from
  **42.7% → 21.9%** — a **~49% relative reduction** — with the fine-tuned model
  running **on-device in the app**, same footprint as the base model, no cloud.

## The task: dictation that excludes people

Screamer transcribes speech to text entirely on-device: hold a key, talk, and
the text lands wherever you're typing. On-device matters — it's free at the
margin, private (audio never leaves the machine), and low-latency.

But "works well" is measured on typical speech. People with **dysarthria** —
motor-speech disorders from cerebral palsy, ALS, stroke, and other conditions —
are exactly the users who would benefit most from hands-light dictation, and
they're the ones general ASR serves worst. We wanted to know: can we close that
gap with a fine-tune, and keep it running on-device?

## Why this is hard

**Dysarthric speech is acoustically atypical** — slurred articulation, irregular
timing, reduced intelligibility — and general ASR is trained overwhelmingly on
typical speech. Off-the-shelf Whisper's error rate on dysarthric audio is high
enough to be unusable for the most affected speakers.

**And it has to run on-device.** Screamer deploys Whisper through whisper.cpp in
the **GGML** format, not a cloud API. So a win isn't just "a better checkpoint" —
it has to survive conversion to GGML and run in the app's own inference stack at
the same size and speed as the model it replaces.

## The open model and the data

- **Base model: `openai/whisper-base.en`** — open weights, and the exact model
  Screamer already ships (`ggml-base.en`). Fine-tuning it keeps the on-device
  footprint identical (~148 MB) and the swap drop-in.
- **Dataset: TORGO** (University of Toronto), a public corpus of dysarthric and
  control speech. We used the dysarthric-female set — three speakers spanning the
  severity range: **F01 (severe), F03 (moderate), F04 (mild)** — recorded on a
  head-mounted mic at 16 kHz.

TORGO mixes verbatim prompts (words and sentences to read aloud) with non-speech
tasks (describe-the-picture prompts, instructions). We built a manifest that
**filters out the non-verbatim prompts** and pairs each remaining clip with its
reference transcript, then split it **utterance-level** — each speaker appears in
training, with a held-out fraction of their utterances reserved for evaluation
(**897 train / 181 held-out**). This measures *adaptation* to these speakers'
speech, the realistic on-device personalization setting.

## Method: supervised fine-tuning on a managed GPU

Standard HuggingFace `transformers` recipe:

- `WhisperProcessor` for log-mel features + label tokenization, a
  speech-seq2seq data collator, and `Seq2SeqTrainer` with
  `predict_with_generate` and a **jiwer-based WER** `compute_metrics`.
- 8 epochs, lr 1e-5, fp16, `whisper-base.en`.

We ran it on **Modal** — serverless GPU, per-second billing, scale-to-zero. The
job is tiny (a few hundred utterances, a T4, well under an hour), so it ran
inside Modal's free monthly credits — **effectively $0**. The run evaluates the
base model first, trains, then evaluates again, so base-vs-tuned comes out of one
command.

## Evaluation

We measure **WER** (via jiwer) on the held-out utterances the model never trained
on, after light normalization (lowercase, strip punctuation). Because TORGO
spans severities, the baseline is naturally **severity-stratified** — a useful,
honest picture of where general ASR breaks down:

| Speaker (severity) | Base Whisper WER |
|---|---|
| F01 (severe) | ~98% |
| F03 (moderate) | ~60% |
| F04 (mild) | ~25% |

Base Whisper is essentially unusable for the most affected speaker and shaky for
the moderate one. A separate CPU baseline (faster-whisper over the full set)
corroborated the picture at **50.3%** overall.

## Results: error roughly halved

On the 181 held-out dysarthric utterances:

| | WER |
|---|---|
| **Base Whisper** (whisper-base.en) | **42.7%** |
| **Fine-tuned** | **21.9%** |
| **Relative reduction** | **~49%** |

Fine-tuning on a few hundred dysarthric utterances **roughly halved the error
rate** on held-out speech from the same speakers. The gains concentrate in the
shorter word- and phrase-length utterances the model adapted to; severe-speaker
*long sentences* remain hard, which is the honest frontier here.

## Deployment: it runs in the app

The point was never a checkpoint on a leaderboard — it was a model a real user
can select. So we took it the last mile:

1. **GGML conversion** with whisper.cpp's `convert-h5-to-ggml.py` (run where the
   model and torch already lived, on Modal). Two known pitfalls handled: the
   fine-tuned config's `max_length` is `null` (set to 448), and the model saves
   as safetensors (re-saved as a torch `.bin` the converter expects).
2. **Bundled into Screamer.app** as a selectable **"Accessibility"** model
   (`ggml-accessibility.en.bin`), same 148 MB footprint as the base.
3. **Verified in the app's own stack** — it loads and transcribes TORGO
   dysarthric audio through Screamer's `whisper-rs` inference, producing output
   distinct from (and, on adapted utterances, better than) base.

So the fine-tuned model runs **fully on-device**: free, private, and selectable
in a shipping app.

## Lessons from the run

- **Decode WAV yourself to dodge dependency skew.** `faster-whisper`'s audio path
  broke against the installed PyAV; reading the 16 kHz/mono/16-bit WAVs directly
  with `wave` + numpy sidestepped it entirely. Likewise, `numpy<2` collided with
  a pyarrow that needs 2.0, and the trainer pulled bleeding-edge `transformers 5.x`
  — both fixed by pinning.
- **GGML conversion has sharp edges.** The `max_length: null` config and the
  safetensors-vs-`.bin` mismatch each silently break `convert-h5-to-ggml.py`;
  both are easy to fix once you know.
- **Managed GPUs still gate on a card.** Modal requires a payment method before
  any GPU runs, even when free credits cover the job.
- **Utterance-split vs speaker-disjoint is a real choice.** Holding out a severe
  speaker *entirely* (generalization to an unseen speaker) is a far harder test
  than holding out their utterances (adaptation); we report the latter and name
  it.

## Why it matters

- **Accessibility.** Halving WER moves on-device dictation from *unusable* toward
  *usable* for people with dysarthria — a population general ASR serves worst, on
  a tool that's free and private by design.
- **Cost.** On-device inference is $0 at the margin with no subscription, and the
  fine-tune itself cost ~$0 on free managed-GPU credits.
- **Privacy.** A dysarthric user's speech — sensitive, identifying — never leaves
  their device.

## Honest caveats and what's next

This is an **experiment**, not a scaled deployment: a single TORGO subset (three
speakers), **SFT only** (no RL), and an **utterance-split** (speaker adaptation,
not unseen-speaker generalization). Natural next steps: the full TORGO set + a
speaker-disjoint eval for the harder generalization claim; whisper-small for more
capacity; and — given NVIDIA's co-hosting and its pending acquisition of Hugging
Face — re-running the identical `transformers` training on a HuggingFace GPU for
an explicit open-stack framing. The training code is already pure HuggingFace;
only the GPU host would change.

## Reproducibility

Everything is in the repo: `evals/torgo_wer.py` (WER scorer),
`evals/torgo_manifest.py` (split/manifest), `evals/train_whisper_modal.py`
(Modal SFT + GGML conversion), `crates/screamer-whisper/src/bin/asr_try.rs`
(in-app spot-check), and `bundle.sh` (ships the model into the app). The TORGO
corpus is public from the University of Toronto; the model binary is not
committed.
