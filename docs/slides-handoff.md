# Screamer — Slide Deck Handoff (for Google Slides)

Paste this into Google Slides (or Gemini "help me create") to generate the deck,
then refine. 7 slides, ~120-second talk track. Each slide below has: **on-slide
text**, **chart/visual**, and **speaker notes** (what you say).

## Branding
- **Logo:** the gold Screamer "scream" mark — `docs/screamer-logo.png` in the repo
  (put it top-left on the title slide, and small in the footer elsewhere).
- **Palette:** cream background `#F9F7F5`, dark-maroon text `#280016`, coral accent
  `#F98461` / `#C2410C`, green for "good/after" `#1D7A4F`. (Matches the Fireworks
  event look; swap to Screamer gold `#E8C547` for accents if you want it more
  on-brand.)
- **Fonts:** a serif for headlines (Crimson Text / Playfair / Georgia), a clean
  sans for body (Inter / Arial), mono for labels (optional).
- **Numbers:** all below are **real, measured** results. Keep them exact.

---

## Slide 1 — Title
**On slide:**
- [Screamer logo] **Screamer**
- *Adaptive, on-device dictation — that works for the people mainstream AI leaves behind.*
- Built for **Project Possibility** (non-profit: software for people with disabilities)
- Ankit Tandon · PM, Intuit  |  Sunanda · PM, Snowflake

**Speaker notes:** "Hi, I'm Ankit, a PM at Intuit, and this is Sunanda, a PM at
Snowflake. We're building for Project Possibility, a non-profit that makes software
for people with disabilities."

---

## Slide 2 — What it is + demo
**On slide:**
- Headline: **Like Wispr Flow — but local, and it adapts to you**
- It runs entirely on your laptop, on on-device models. Adapts to *how you speak* and *how you write*.
- Hold **left-control** → listen, transcribe, style. Instantly. Nothing leaves your machine.

**Visual:** a screenshot/GIF of Screamer in action (or the live demo).

**Speaker notes:** "We built Screamer — think Wispr Flow, but it runs entirely on
your laptop on on-device models, and it adapts to how you speak and write. Watch —
I hold left-control, and it listens, transcribes, and styles, instantly." *[dictate a line]*

---

## Slide 3 — The problem + business value
**On slide (3 stat tiles):**
- Mainstream: **2–3%** WER (frontier models)
- Speech disability: **40–60%** WER (out-of-the-box)
- Non-native English: **10%+** WER — **1B+** speakers
- Caption: Transcription + styling that *just works* here is worth billions —
  Microsoft bought **Nuance ~$20B**; **Abridge ~$3B**. **Screamer can adapt to be that.**

**Speaker notes:** "For mainstream English, recognition is basically solved —
2–3% error. But at the edges it breaks: 40–60% error for people with speech
impairments, and 10%-plus for over a billion non-native speakers. Getting this
right is worth billions — Microsoft bought Nuance for ~$20B, Abridge is near $3B.
Screamer can adapt to be exactly that."

---

## Slide 4 — THE MONEY SLIDE: the ASR win
**On slide:** headline **Open models, fine-tuned, beat frontier — on the people they fail**

**Chart (grouped bar, lower = better, "Base" dark vs "Fine-tuned" coral):**

| Dataset (held-out) | Base WER | Fine-tuned WER |
|---|--:|--:|
| Dysarthric — TORGO (female) | 42.7 | 22.3 |
| Dysarthric — TORGO (male) | 161.6 | 33.5 |
| Non-native/accented — L2-ARCTIC | 14.9 | 5.7 |

- Sub-caption: **Statistically significant** — bootstrap + McNemar (p as low as 10⁻¹³).
  Small model, in **Rust**, **on-device**: free, private, fast.
- Footnote: base >100% = the model hallucinates more words than were spoken on severe speech.

**Speaker notes:** "We fine-tuned open Whisper models on three datasets — dysarthric
speech from TORGO, and accented non-native English from L2-ARCTIC. We cut error
dramatically on every one, and it's statistically significant on two tests,
bootstrap and McNemar. It's a small model, in Rust, on-device — so it's free,
private, and fast."

---

## Slide 5 — The styling win
**On slide:** headline **It writes like you — not like AI**

**Chart (grouped bar — our styling vs a frontier model's out-of-box output):**

| Metric | Frontier, naive prompt | Ours (Fireworks + harness) |
|---|--:|--:|
| Slop / 1k (lower better) | 27.5 | 8.4 |
| Intent fidelity (0–1) | 0.92 | 0.98 |
| Writing quality (0–1) | 0.87 | 0.93 |

- Bullets: Rewrites raw speech into clean writing in your voice, stripping AI "slop."
  Built on **Fireworks** + its **Eval Protocol**; **beats the frontier model's
  out-of-box output on every metric, significantly** (McNemar p=1.4×10⁻⁵).
- And it learns from **you**: keep/revert feedback personalizes it on-device, from
  a handful of your own corrections.

**Speaker notes:** "Then we added a styling layer — an LLM that rewrites your raw
transcript into clean writing in your own voice, without the AI 'slop.' On
Fireworks, it beats a frontier model's out-of-the-box output on slop, intent, and
quality — all significant. And it learns from you: every keep or revert
personalizes it, on-device."

---

## Slide 6 — Architecture
**On slide:** headline **Two layers of adaptation, one private flywheel**
- **The ear** — ASR, tuned to how you speak · on-device · NVIDIA GPUs
- **The pen** — styling, tuned to how you write · Fireworks
- **The flywheel** — per-user, on-device, your data never leaves
- Stack: **Rust · GGML · Metal** (on-device) · **Fireworks** (styling + Eval Protocol) · **NVIDIA** (open models + inference)

**Visual:** the two-layer + flywheel diagram (see `docs/case-study` / the architecture SVG).

**Speaker notes:** "It all fits together as two layers — the ear and the pen — with
an on-device flywheel that keeps improving both, per-user, privately."

---

## Slide 7 — Close
**On slide:** headline **Raise the floor — not just the ceiling**
- Most of AI races to raise the ceiling. Specialized, **open** intelligence can
  raise the floor — making AI work for the people the frontier leaves behind.
- That's the beauty of open source: intelligence anyone can **own and adapt**.
- **Accessibility for all.**

**Speaker notes:** "Here's our hope. Most of the AI race is about raising the
ceiling. We want to raise the floor — specialized, open models that make AI work
for the people it leaves behind. That's the beauty of open source: intelligence
anyone can own and adapt. Accessibility for all. Thank you."

---

## Notes on honesty (keep the deck credible)
- Styling result is **deepseek-flash on Fireworks** (the strong, significant win).
  We *tried* fine-tuning an NVIDIA model for styling; the small Nemotron-4B wasn't
  good enough (runaway output) — so we don't claim it. The NVIDIA story is: **open
  models fine-tuned on NVIDIA GPUs** + the **Parakeet-via-NIM baseline (60.3% on
  dysarthric)** we measured as the frontier comparison.
- Every number here is in the repo and reproducible (`evals/`).
