"""Eval Protocol suite for the Screamer styling layer.

These score *stored* dictation traces — they do not call a model. The
`NoOpRolloutProcessor` passes each row straight through to the scorer, so the
assistant message is the text the styling layer actually pasted.

Dataset: an `ExportRow` JSONL from `screamer-telemetry-export`. Point at your own
export with the `SCREAMER_EVAL_DATASET` env var; otherwise the committed
synthetic sample is used so the suite runs anywhere.

    cd evals && pip install -e . && pytest
"""

# NB: no `from __future__ import annotations` here — Eval Protocol inspects the
# decorated function's real `row: EvaluationRow` annotation, which stringified
# annotations would break.

import os
from pathlib import Path
from typing import Any, Dict

from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor

from adapter import screamer_dataset_adapter
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


def _dataset_info(row: EvaluationRow) -> Dict[str, Any]:
    return (row.input_metadata.dataset_info if row.input_metadata else None) or {}


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
# Scaffolds. The two metrics that matter most — did the rewrite keep the
# speaker's meaning, and is it actually better writing — need an LLM judge, not
# a rules pass. These are real EP tests wired to the same data so the harness
# shape is complete; they emit a placeholder, mark the score invalid, and never
# gate CI (passed_threshold=0.0) until the judge lands.
# ---------------------------------------------------------------------------


@evaluation_test(
    input_dataset=[DATASET],
    dataset_adapter=screamer_dataset_adapter,
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
)
def test_intent_fidelity(row: EvaluationRow) -> EvaluationRow:
    """SCAFFOLD — does the styled text preserve the speaker's meaning and intent?

    The real check is an LLM judge comparing raw (user) vs styled (assistant):
    did the rewrite add, drop, or alter a claim; turn a question into a
    statement; harden a hedge; resolve an ambiguity the speaker left open? See
    the intent discussion in docs/styling-guide.md.

    TODO: replace the placeholder below with a Fireworks LLM-judge call. For now
    it is marked invalid so it never counts toward pass/fail.
    """
    edit_ratio = _dataset_info(row).get("edit_ratio")
    placeholder = 1.0 - float(edit_ratio) if isinstance(edit_ratio, (int, float)) else 1.0
    row.evaluation_result = EvaluateResult(
        score=placeholder,
        is_score_valid=False,
        reason="scaffold: intent fidelity needs an LLM judge (not yet implemented)",
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
    """SCAFFOLD — is the styled text genuinely better writing (clearer, tighter)?

    Beyond "not sloppy": concision, clarity, and whether it reads like something
    the speaker would have written with time. This also wants an LLM judge.

    TODO: replace with an LLM-judge call. Marked invalid until then.
    """
    styled = _text(row, "assistant")
    row.evaluation_result = EvaluateResult(
        score=_slop_to_score(slop_score(styled)),  # placeholder proxy only
        is_score_valid=False,
        reason="scaffold: writing-quality judge not yet implemented",
    )
    return row
