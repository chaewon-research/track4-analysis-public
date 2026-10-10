"""Corpus indexer: one retrievable chunk per corpus span, with global offsets.

Corpus documents carry their text as a ``spans`` array (or a flat ``text``
field). The scorer's offset convention concatenates span texts with a single
space, so a span that starts at position ``p`` in that concatenation can be
cited directly as ``(doc_id, p, p + len(span_text))``. Indexing at span
granularity therefore gives every chunk an exact, citation-ready offset pair
for free — no separate span search needed for chunk-level citations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    doc_date: str | None
    span_start: int  # global char offset into the document's joined text
    span_end: int
    text: str


@dataclass(frozen=True)
class IndexedCorpus:
    chunks: list[Chunk]
    doc_texts: dict[str, str]  # doc_id -> full joined text (for span finding)
    doc_dates: dict[str, str | None]
    # Manifest-declared document/entity bindings. Empty for corpora whose
    # manifest does not declare entity_ids.
    doc_entity_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    shared_doc_ids: frozenset[str] = field(default_factory=frozenset)


def _iter_span_texts(doc: dict) -> list[str]:
    if isinstance(doc.get("text"), str):
        return [doc["text"]]
    spans = doc.get("spans")
    if isinstance(spans, list):
        return [sp.get("text", "") for sp in spans if isinstance(sp, dict)]
    return []


def build_index(corpus_dir: str | Path) -> IndexedCorpus:
    """Index every corpus document under ``corpus_dir`` (skips the manifest)."""
    chunks: list[Chunk] = []
    doc_texts: dict[str, str] = {}
    doc_dates: dict[str, str | None] = {}
    doc_entity_ids: dict[str, frozenset[str]] = {}
    shared_doc_ids: set[str] = set()

    corpus_path = Path(corpus_dir)

    # The public corpus manifest is the authoritative document/entity binding
    # when it supplies entity_ids. Keep it with the index so retrieval can
    # avoid giving one roster entity another entity's document.
    entity_ids_by_stem: dict[str, frozenset[str]] = {}
    shared_stems: set[str] = set()
    manifest_path = corpus_path / _MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("files", []):
            if not isinstance(item, dict):
                continue
            raw_path = item.get("path")
            if not isinstance(raw_path, str):
                continue

            stem = Path(raw_path).stem

            raw_entity_ids = item.get("entity_ids")
            if isinstance(raw_entity_ids, list):
                ids = frozenset(
                    str(entity_id)
                    for entity_id in raw_entity_ids
                    if isinstance(entity_id, str) and entity_id
                )
                if ids:
                    entity_ids_by_stem[stem] = ids

            if item.get("shared") is True:
                shared_stems.add(stem)

    for path in sorted(corpus_path.glob("*.json")):
        if path.name == _MANIFEST_NAME:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc_id = doc.get("doc_id", path.stem)
        doc_date = doc.get("doc_date")
        offset = 0
        parts: list[str] = []
        for text in _iter_span_texts(doc):
            if text:
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        doc_date=doc_date,
                        span_start=offset,
                        span_end=offset + len(text),
                        text=text,
                    )
                )
            parts.append(text)
            offset += len(text) + 1  # +1 for the joining space
        doc_texts[doc_id] = " ".join(parts)
        doc_dates[doc_id] = doc_date

        manifest_ids = entity_ids_by_stem.get(path.stem)
        if manifest_ids:
            doc_entity_ids[doc_id] = manifest_ids
        if path.stem in shared_stems:
            shared_doc_ids.add(doc_id)

    return IndexedCorpus(
        chunks=chunks,
        doc_texts=doc_texts,
        doc_dates=doc_dates,
        doc_entity_ids=doc_entity_ids,
        shared_doc_ids=frozenset(shared_doc_ids),
    )
