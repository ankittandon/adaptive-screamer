# Fine-tuning the styling model

Turn your stored dictation traces into a Fireworks supervised fine-tuning (SFT)
dataset, then run the training job. The goal is to teach the styling model
`raw transcript → the final text you actually kept`.

## 1. Export the dataset

```bash
cargo run -p screamer-styling --bin finetune-export -- \
  "$HOME/Library/Application Support/Screamer/telemetry" > sft.jsonl
```

Each line is one example in Fireworks' chat SFT format:

```json
{"messages":[{"role":"system","content":"<styling prompt>"},{"role":"user","content":"<raw transcript>"},{"role":"assistant","content":"<final settled text>"}]}
```

The system message is the current `SYSTEM_PROMPT` from `screamer-styling`, so
training matches inference. Only the assistant turn is trained (Fireworks masks
system/user from the loss). The exporter:

- keeps only labeled rows (you corrected or the implicit watcher captured a final),
- drops rows whose correction rewrote more than `--max-edit-ratio` of the words
  (default 0.8 — likely a bad label),
- skips empty transcripts and de-duplicates identical `(raw, final)` pairs.

Flags: `--no-system-prompt` (train prompt-free, to shrink the inference prompt),
`--max-edit-ratio F`. The written/skipped counts print to stderr; Fireworks SFT
needs at least 3 examples.

## 2. Create the dataset on Fireworks and launch the job

`firectl` refuses mutating commands when run by an AI agent, so **run these in
your own terminal.** Install it once with `brew install fw-ai/firectl/firectl`,
and make sure your key is available:

```bash
export FIREWORKS_API_KEY="$(cat "$HOME/Library/Application Support/Screamer/fireworks.key")"
```

Create the dataset (not billed — just uploads the data):

```bash
firectl create dataset screamer-styling-sft sft.jsonl
```

Launch the fine-tune (this is the billed step, and it creates a model on your
account):

```bash
firectl create sftj --base-model <tunable-base-model> --dataset screamer-styling-sft --output-model screamer-styling-v1
```

Pick a Fireworks-tunable base model for `--base-model`; the current inference
model (`deepseek-v4p1-flash`) may not itself be fine-tunable. Monitor with
`firectl get sftj <job-id>`, then deploy and point `FIREWORKS_MODEL` at the
resulting model to use it in the app.

## Notes and next steps

- **Thin data.** With a few dozen labeled rows this validates the pipeline more
  than it produces a strong model. It becomes useful as traces accumulate.
- **Identity-heavy signal.** Many labels (especially implicit captures) have a
  final that nearly equals the raw transcript, so a plain SFT run leans toward
  "copy." A useful refinement is to weight toward rows where the final
  meaningfully differs from the raw, or to filter on the intent/quality judge
  scores (see `evals/`).
- **DPO later.** Preference tuning (chosen = your final text, rejected = the
  first styled paste where they differ) is the natural next step once Fireworks'
  preference line format is confirmed.
