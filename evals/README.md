# Screamer styling evals

An [Eval Protocol](https://evalprotocol.io) suite that scores the styling
layer's output. The evals score **stored** dictation traces — they do not call a
model — so they run offline and deterministically.

## Layout

- `scorers/slop.py` — Python port of the Rust slop detector
  (`crates/screamer-styling/src/slop.rs`). Keep the two in sync.
- `adapter.py` — turns a telemetry `ExportRow` into an EP `EvaluationRow`
  (`user` = raw transcript, `assistant` = pasted/styled text, `ground_truth` =
  settled text).
- `test_styling.py` — the EP tests:
  - `test_styled_slop` — real, offline: the pasted text should read human
    (low slop). Gated at a lenient 0.5.
  - `test_intent_fidelity`, `test_quality` — **scaffolds**. The two metrics that
    matter most need an LLM judge; these are wired to the data and emit an
    invalid placeholder (never gate CI) until the judge lands. See the TODOs.
- `test_slop_parity.py` — plain pytest (no EP needed) proving the slop port
  matches the Rust unit tests. The hermetic CI gate.
- `data/sample.jsonl` — synthetic traces so the suite runs anywhere.

## Run

```bash
cd evals
pip install -e .
pytest                      # whole suite
pytest test_slop_parity.py  # just the slop port check (no EP install needed)
```

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
