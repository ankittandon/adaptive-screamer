# Demo video script (target ~55s, hard cap 120s)

Rubric wants three things: **explain the problem, demonstrate the approach, show
the measured result.** Keep the baseline→after comparison and the CI on screen.

**Setup to have ready on screen:** Screamer running with the Accessibility model
selectable; one dysarthric TORGO clip queued; the CI table from `SUBMISSION.md` §2
in a tab; the latency/cost table.

| Time | On screen | Say (voiceover) |
|---|---|---|
| 0:00–0:10 | Play a severe dysarthric clip → base Whisper transcribes it **wrong** | "Dictation tools don't work for people with dysarthric speech. Here's general Whisper on a real clip — it fails." |
| 0:10–0:22 | Same clip in Screamer, switch to the **Accessibility** model → correct text appears, on-device | "We fine-tuned open Whisper on public dysarthric data, converted it to GGML, and it runs on-device in our shipping app. Same clip — now it's right." |
| 0:22–0:38 | Show the three-split CI table | "And it's not memorization. On words it never trained on, WER drops 31.5 to 21.5 — ten points, 95% CI 5.3 to 15.2, p under 0.001. On a severe voice it never heard, 135 down to 63. Both statistically significant." |
| 0:38–0:48 | Latency/cost table; then a flash of the Eval Protocol / Fireworks styling | "On-device is 2.3× faster than cloud at zero cost and fully private. The styling layer and our evaluation harness run on Fireworks' Eval Protocol." |
| 0:48–0:58 | App transcribing live; end on the title | "Specialized open models beating frontier ASR — free, private, and finally usable by the people it used to fail. And it ships." |

## Tips
- Lead with the **failure then the fix** — the before/after is the whole story.
- Keep the **CI numbers visible** while you say them; judges weight evidence (40%).
- Say "Fireworks" out loud (eligibility) and "NVIDIA Parakeet / NIM" once (track).
- If you run long, cut the efficiency line (0:38–0:48) first — accessibility +
  evidence is the core.
- Record at 1080p; show the terminal/numbers large enough to read.
