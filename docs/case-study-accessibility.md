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
  **42.7% → 22.3%** — a **~48% relative reduction**, and the error roughly halved
  across *every* severity stratum (severe speaker 89.7% → 44.8%). The fine-tuned
  model runs **on-device in the app**, same footprint as the base, no cloud.

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

We measure **WER** (via jiwer) on the 181 held-out utterances the model never
trained on, decoded with HuggingFace `generate`, after light normalization
(lowercase, strip punctuation). Because TORGO spans severities, we report the
result **stratified by speaker** — the honest picture of where general ASR breaks
down and where the fine-tune helps most. (An independent CPU baseline with
faster-whisper over the full F set corroborated that base degrades sharply with
severity — ~50% WER overall, far worse for the severe speaker.)

## Results: error roughly halved — most where it mattered most

Base vs fine-tuned WER on the held-out split, by severity stratum (lower is
better):

| Stratum | n | Base WER | Fine-tuned WER | Δ (pts) | Relative |
|---|---:|---:|---:|---:|---:|
| **F01 (severe)** | 19 | 89.7% | **44.8%** | −44.8 | **−50%** |
| **F03 (moderate)** | 91 | 52.4% | **31.2%** | −21.2 | **−40%** |
| **F04 (mild)** | 71 | 22.5% | **7.0%** | −15.5 | **−69%** |
| **Overall** | 181 | **42.7%** | **22.3%** | −20.4 | **−48%** |

Fine-tuning on a few hundred dysarthric utterances **cut the overall error rate
roughly in half** — and the gains hold across the severity range. The two
headline moves:

- The **severe** speaker (F01) went from **89.7% → 44.8%** — from effectively
  unusable toward partially usable, the largest absolute gain (−44.8 pts).
- The **mild** speaker (F04) reached **7.0%** WER — genuinely usable dictation,
  the largest *relative* gain (−69%).

The severe-speaker *long sentences* remain the hard frontier; the biggest wins
are on the shorter word- and phrase-length utterances the model adapted to.

## How we know it isn't overfitting

Fine-tuning on TORGO and evaluating on TORGO invites the obvious question: did we
just memorize the test set? We take it seriously, because our first split had two
real shortcuts — **84% of eval sentences also appeared in training** (TORGO has
everyone read the same prompts) and the **same speakers** were on both sides. So
we re-ran against two harder, leakage-free splits, and bootstrapped every number.

**Three held-out splits** (base vs fine-tuned WER; reduction with a 95% bootstrap
CI over 10,000 resamples and a one-sided paired p-value):

| Split — what it tests | n | Base | Tuned | Reduction (95% CI) | p |
|---|--:|--:|--:|---|--:|
| **Utterance** — adaptation to a known user | 181 | 42.7% | 22.3% | **−20.4 [14.9, 26.8]** | <0.0001 |
| **Text-disjoint** — words never seen in training | 210 | 31.5% | 21.5% | **−10.0 [5.3, 15.2]** | <0.0001 |
| **Speaker-disjoint** — a severe voice never heard | 114 | 135.6% | 62.9% | **−72.7 [11.6, 194.5]** | <0.0001 |

Every reduction CI excludes zero, and the gain holds **per severity stratum** on
the text-disjoint split too (severe −28.8, moderate −7.3, mild −8.4, all p < 0.02).
So the win survives removing both shortcuts: the model transcribes **words it
never trained on** and a **voice it never heard** — it learned dysarthric
acoustics, not TORGO's vocabulary. (The reduction is a *paired* statistic — base
and tuned are scored on the same clips — so it's estimated tightly even though
absolute dysarthric WER is high-variance clip to clip.)

Supporting guards: eval clips are excluded from training; the checkpoint is
selected by held-out WER (`load_best_model_at_end`), not train loss; and small LR
/ few epochs / encoder-freeze limit memorization. And in the continuous-learning
loop, a **validation gate** promotes a new adapter only if held-out WER improves —
so the product structurally refuses to ship a model that overfit.

Match the eval to the claim: for **personalization** (adapting to one user's voice
and vocabulary) the utterance split is the *correct* test and −48% relative is
valid; for the broader **"we improved dysarthric ASR"** claim, the text- and
speaker-disjoint numbers are the honest ones.

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
