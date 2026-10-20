"""Turn Screamer telemetry export rows into Eval Protocol `EvaluationRow`s.

Input is the JSONL produced by `screamer-telemetry-export` (one `ExportRow` per
line: raw_text, pasted_text, settled_text, edit_ratio, app_context, ...). We lay
each dictation out as a two-message conversation so scorers can read it with the
standard EP shape:

    user      -> raw_text    (what Whisper produced)
    assistant -> pasted_text (what the styling layer pasted)
    ground_truth -> settled_text (what the user ultimately kept, if labeled)

The remaining signals (app, edit_ratio, label source, model) ride in
`input_metadata.dataset_info` for scorers that want them.
"""

from __future__ import annotations

from typing import Any, Dict, List

from eval_protocol.models import EvaluationRow, InputMetadata, Message


def screamer_dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    converted: List[EvaluationRow] = []
    for r in rows:
        raw = r.get("raw_text") or ""
        pasted = r.get("pasted_text") or ""
        settled = r.get("settled_text")  # may be None (unlabeled)
        converted.append(
            EvaluationRow(
                messages=[
                    Message(role="user", content=raw),
                    Message(role="assistant", content=pasted),
                ],
                ground_truth=settled,
                input_metadata=InputMetadata(
                    row_id=r.get("id"),
                    dataset_info={
                        "app_context": r.get("app_context"),
                        "edit_ratio": r.get("edit_ratio"),
                        "label_source": r.get("label_source"),
                        "model": r.get("model"),
                    },
                ),
            )
        )
    return converted
