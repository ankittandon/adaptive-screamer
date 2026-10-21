# Screamer styling evals

An [Eval Protocol](https://evalprotocol.io) suite that scores the styling
layer's output. The evals score **stored** dictation traces — they do not call a
model — so they run offline and deterministically.

## Layout

- `scorers/slop.py` — Python port of the Rust slop detector
  (`crates/screamer-styling/src/slop.rs`). Keep the two in sync.
- `scorers/judge.py` — Fireworks LLM judges for intent fidelity and writing
  quality (temperature 0, returns a 0..1 score + reason).
- `adapter.py` — turns a telemetry `ExportRow` into an EP `EvaluationRow`
  (`user` = raw transcript, `assistant` = pasted/styled text, `ground_truth` =
  settled text).
- `test_styling.py` — the EP tests:
  - `test_styled_slop` — rules-based, offline: the pasted text should read human
    (low slop). Gated at a lenient 0.5.
  - `test_intent_fidelity` — LLM judge: did the rewrite keep the speaker's
    meaning and intent (claims, speech act, hedging)?
  - `test_quality` — LLM judge: is the styled text genuinely good writing
    (clear, tight, not AI-sloppy)?
  - The two judges run only when a Fireworks key is available; with none the
    rows come back invalid (the run still completes). They are informational for
    now (`passed_threshold=0.0`) — the scores are real, but we haven't gathered
    enough to set a meaningful gate.
- `test_slop_parity.py` — plain pytest (no EP needed) proving the slop port
  matches the Rust unit tests. The hermetic CI gate.
- `data/sample.jsonl` — synthetic traces so the suite runs anywhere.

## Run

```bash
cd evals
pip install -e .
pytest                          # whole suite (judges run if a Fireworks key is set)
pytest test_slop_parity.py      # just the slop port check (no EP install needed)
pytest test_styling.py::test_styled_slop   # offline eval only (what CI runs)
```

The LLM judges read a Fireworks key from `FIREWORKS_API_KEY`, or the app's key
file at `~/Library/Application Support/Screamer/fireworks.key`. Override the
judge model with `FIREWORKS_JUDGE_MODEL`. CI runs only the offline slop eval; the
judges call Fireworks (key + network + cost), so they run locally. Note that
judging a real export sends that dictation text to Fireworks — the same provider
the styling layer already uses; the committed sample is synthetic.

## Point it at your real data

Export your telemetry to the `ExportRow` JSONL the suite reads, then set the
dataset path:

```bash
cargo run -p screamer-telemetry --bin screamer-telemetry-export -- \
  "$HOME/Library/Application Support/Screamer/telemetry" --all > my_traces.jsonl
SCREAMER_EVAL_DATASET="$PWD/my_traces.jsonl" pytest
```

Braintrust is the trace **store/browser** (see `src/braintrust.rs`); these evals
read the local export so they stay offline. Pushing eval scores back to
Braintrust alongside the traces is a later follow-up.
