"""Prompt construction for the strong RAG baseline.

One prompt per entity. The model sees the entity's tabular features and the
top-K retrieved excerpts, each tagged with its ``doc_id``, and must return a
single JSON object. Evidence quotes are required to be verbatim substrings of
the provided excerpts — the span finder maps them back to exact character
offsets, and anything it cannot locate is dropped rather than cited loosely.
"""
from __future__ import annotations

import json

from .indexer import Chunk

SYSTEM_PROMPT = """\
You are a careful financial analyst. You predict a target for one entity using ONLY the
evidence excerpts provided — no outside knowledge about events after the stated cutoff date.
You must respond with a single JSON object and nothing else. Every evidence quote you return
must be copied verbatim, character for character, from one of the provided excerpts."""

_TARGET_INSTRUCTIONS = {
    "classification": (
        'Set "label" to exactly one of the allowed labels. '
        'Set "point_forecast" to your best numeric estimate of the underlying quantity '
        "if one is defined for this task, else null."
    ),
    "regression": (
        'Set "point_forecast" to your numeric prediction of the target. '
        '"label" may be null.'
    ),
    "ranking": (
        'Set "point_forecast" to the predicted metric value; entities are ranked by it. '
        '"label" may be null.'
    ),
}


def build_user_prompt(
    task: dict, entity: dict, retrieved: list[Chunk]
) -> str:
    target = task.get("target", {})
    target_type = target.get("type", "classification")
    lines: list[str] = []

    lines.append(f"TASK: {task.get('prompt', '')}")
    lines.append(f"CUTOFF DATE: {task.get('cutoff_date', '')}")
    lines.append(f"TARGET: {target.get('name', '')} ({target_type})")
    if target.get("labels"):
        lines.append(f"ALLOWED LABELS: {', '.join(target['labels'])}")
    interval_level = task.get("interval_level", 0.90)

    lines.append("\nENTITY:")
    for key, value in entity.items():
        if key in ("corpus_ref",):
            continue
        lines.append(f"  {key}: {value}")

    lines.append("\nEVIDENCE EXCERPTS (cite only these):")
    for i, chunk in enumerate(retrieved, 1):
        lines.append(f"[{i}] doc_id={chunk.doc_id} (doc_date={chunk.doc_date})")
        lines.append(f'"""{chunk.text}"""')

    # Keep the requested output contract extremely explicit. Smaller local
    # models in particular can otherwise copy exhibit numbers such as "10.1"
    # as document IDs or treat required classification fields as optional.
    if target_type in ("regression", "ranking"):
        point_instruction = "REQUIRED finite number; never null"
    elif target_type == "classification":
        point_instruction = (
            "number if the TASK requests a numeric point_forecast/probability; "
            "otherwise null"
        )
    else:
        point_instruction = "number or null"

    schema = {
        "label": (
            "EXACTLY one ALLOWED LABEL; never null"
            if target_type == "classification"
            else "string or null"
        ),
        "point_forecast": point_instruction,
        "interval": {
            "level": interval_level,
            "lo": "number",
            "hi": "number",
        },
        "evidence": [
            {
                "doc_id": "copy an exact doc_id shown in EVIDENCE EXCERPTS",
                "quote": "verbatim substring copied from that same excerpt",
                "claim": "one factual sentence directly supported by the quote",
            }
        ],
    }
    lines.append(
        "\nRespond with ONE JSON object of this shape (no markdown fences, no prose):"
    )
    lines.append(json.dumps(schema, indent=2))
    lines.append(f"\n{_TARGET_INSTRUCTIONS.get(target_type, _TARGET_INSTRUCTIONS['classification'])}")
    valid_doc_ids = ", ".join(dict.fromkeys(c.doc_id for c in retrieved))
    lines.append(
        "\nSTRICT OUTPUT RULES:\n"
        "- Never return null for label on a classification task.\n"
        "- For regression and ranking tasks, point_forecast MUST be a finite "
        "number and MUST NEVER be null.\n"
        "- Copy doc_id EXACTLY from an EVIDENCE EXCERPT header. Do not add "
        '"doc_id=", an excerpt number, an exhibit number, or any other prefix.\n'
        "- Return exactly ONE evidence entry: the strongest directly supported "
        "pre-cutoff fact relevant to your prediction.\n"
        "- Choose the evidence quote FIRST, then write the claim as a conservative "
        "restatement of facts explicitly stated in that quote.\n"
        "- The evidence quote must be copied verbatim from that same document.\n"
        "- Do NOT add a number, date, direction, causal relation, or factual conclusion "
        "to the claim unless that information is explicitly stated in the quote.\n"
        "- If the claim contains a number or date, that same number or date MUST appear "
        "verbatim in the evidence quote.\n"
        "- Do NOT use an ENTITY field itself as evidence unless the same fact also "
        "appears in the quoted EVIDENCE EXCERPT.\n"
        "- Do NOT combine facts from different excerpts into one claim.\n"
        "- Prefer a specific observed pre-cutoff fact over an inferred explanation.\n"
        "- lo must be <= point_forecast <= hi whenever point_forecast is numeric.\n"
        f"- Valid doc_ids for this request: {valid_doc_ids}"
    )
    lines.append(
        f'The "interval" must be your {int(interval_level * 100)}% prediction interval for the '
        "numeric target: wide enough that you expect the realized value to fall inside it "
        f"{int(interval_level * 100)}% of the time, and no wider."
    )

    task_text = str(task.get("prompt", "")).lower()
    if (
        target_type == "classification"
        and "point_forecast" in task_text
        and "probability" in task_text
    ):
        lines.append(
            'This TASK explicitly defines point_forecast as a probability. '
            'Therefore point_forecast MUST be a number from 0 to 1, never null. '
            'The interval must satisfy 0 <= lo <= point_forecast <= hi <= 1.'
        )
    return "\n".join(lines)
