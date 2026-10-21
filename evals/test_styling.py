"""Eval Protocol suite for the Screamer styling layer.

These score *stored* dictation traces — they do not call a model to generate.
The `NoOpRolloutProcessor` passes each row straight through to the scorer, so the
assistant message is the text the styling layer actually pasted.

Dataset: an `ExportRow` JSONL from `screamer-telemetry-export`. Point at your own
export with the `SCREAMER_EVAL_DATASET` env var; otherwise the committed
synthetic sample is used so the suite runs anywhere.

Metrics:
- `test_styled_slop` — rules-based, offline, deterministic (slop detector).
- `test_intent_fidelity`, `test_quality` — Fireworks LLM judges (scorers/judge.py).
  They run only when a Fireworks key is available; with none the rows come back
  invalid so a keyless run (e.g. CI) degrades gracefully instead of failing.

    cd evals && pip install -e . && pytest
"""

# NB: no `from __future__ import annotations` here — Eval Protocol inspects the
# decorated function's real `row: EvaluationRow` annotation, which stringified
# annotations would break.

import os
from pathlib import Path

from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor

from adapter import screamer_dataset_adapter
from scorers.judge import judge_intent, judge_quality
from scorers.slop import slop_score

DATASET = os.environ.get(
    "SCREAMER_EVAL_DATASET",
    str(Path(__file__).parent / "data" / "sample.jsonl"),
)


def _text(row: EvaluationRow, role: str) -> str:
    msgs = [m for m in row.messages if m.role == role]
    content = msgs[-1].content if msgs else ""
    if isinstance(content, list):  # content parts
        content = "".join(getattr(p, "text", "") for p in content)
    return content or ""


def _slop_to_score(slop: float) -> float:
    """Per-1000-words slop -> 0..1 (higher is better).

    0 slop -> 1.0, 15 -> 0.5, >= 30 -> 0.0. Human prose is typically well under
    ~15 (see docs/styling-guide.md), so 0.5 is a lenient floor.
    """
    return max(0.0, 1.0 - slop / 30.0)


@evaluation_test(
    input_dataset=[DATASET],
    dataset_adapter=screamer_dataset_adapter,
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.5,
)
def test_styled_slop(row: EvaluationRow) -> EvaluationRow:
    """The pasted (styled) text should read human, not like AI slop.

    Uses the same detector as the app (`scorers/slop.py`, a port of slop.rs).
    """
    styled = _text(row, "assistant")
    raw = _text(row, "user")
    s_styled = slop_score(styled)
    s_raw = slop_score(raw)
    score = _slop_to_score(s_styled)
    row.evaluation_result = EvaluateResult(
        score=score,
        is_score_valid=True,
        reason=f"styled slop {s_styled:.1f}/1000 words (raw {s_raw:.1f})",
        metrics={
            "slop_styled": MetricResult(
                score=score,
                is_score_valid=True,
                reason=f"{s_styled:.1f} per 1000 words",
                data={"slop_per_1000": s_styled},
            ),
            "slop_raw": MetricResult(
                score=_slop_to_score(s_raw),
                is_score_valid=True,
                reason=f"{s_raw:.1f} per 1000 words",
                data={"slop_per_1000": s_raw},
            ),
        },
    )
    return row


# ---------------------------------------------------------------------------
# LLM judges. The two metrics that matter most — did the rewrite keep the
# speaker's meaning, and is it actually good writing — are scored by a Fireworks
# judge (scorers/judge.py). They run only when a Fireworks key is available; with
# no key the rows come back invalid so a keyless run (e.g. CI) degrades
# gracefully. passed_threshold is 0.0 for now (informational): the scores are
# real, but we have not gathered enough runs to set a meaningful gate — raise it
# once we have.
# ---------------------------------------------------------------------------


@evaluation_test(
    input_dataset=[DATASET],
    dataset_adapter=screamer_dataset_adapter,
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
)
def test_intent_fidelity(row: EvaluationRow) -> EvaluationRow:
    """Does the styled text preserve the speaker's meaning and intent?

    Fireworks judge comparing raw (user) vs styled (assistant). See judge.py.
    """
    raw = _text(row, "user")
    styled = _text(row, "assistant")
    j = judge_intent(raw, styled)
    row.evaluation_result = EvaluateResult(
        score=j.score,
        is_score_valid=j.valid,
        reason=j.reason,
    )
    return row


@evaluation_test(
    input_dataset=[DATASET],
    dataset_adapter=screamer_dataset_adapter,
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
)
def test_quality(row: EvaluationRow) -> EvaluationRow:
    """Is the styled text genuinely good writing (clear, tight, not AI-sloppy)?

    Fireworks judge scoring the styled (assistant) text against the original. See
    judge.py.
    """
    raw = _text(row, "user")
    styled = _text(row, "assistant")
    j = judge_quality(raw, styled)
    row.evaluation_result = EvaluateResult(
        score=j.score,
        is_score_valid=j.valid,
        reason=j.reason,
    )
    return row
