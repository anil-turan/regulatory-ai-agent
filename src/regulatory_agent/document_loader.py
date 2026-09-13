"""Loads the curated regulatory markdown corpus and chunks it by section.

Chunking strategy: split on level-3 (###) headings, which in this corpus
correspond to individual rule/clause-level statements. Each chunk keeps its
parent level-2 (##) section title and source document name as metadata, so
retrieval results can carry a real citation (document + section + clause).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    section: str
    clause: str
    text: str
    chunk_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.chunk_id = f"{self.doc_id}::{self.clause}".replace(" ", "_")

    def citation(self) -> str:
        return f"{self.doc_title} — {self.section} — {self.clause}"


_H1 = re.compile(r"^#\s+(.*)$")
_H2 = re.compile(r"^##\s+(.*)$")
_H3 = re.compile(r"^###\s+(.*)$")


def load_document(path: Path) -> list[Chunk]:
    """Parse one markdown file into a list of clause-level Chunks."""
    doc_id = path.stem
    lines = path.read_text(encoding="utf-8").splitlines()

    doc_title = doc_id
    section = ""
    clause = ""
    buffer: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text and clause:
            chunks.append(
                Chunk(doc_id=doc_id, doc_title=doc_title, section=section, clause=clause, text=text)
            )

    for line in lines:
        if m := _H1.match(line):
            doc_title = m.group(1).strip()
            continue
        if m := _H2.match(line):
            flush()
            buffer = []
            section = m.group(1).strip()
            clause = ""
            continue
        if m := _H3.match(line):
            flush()
            buffer = []
            clause = m.group(1).strip()
            continue
        if line.strip().startswith(">"):
            # source-note callouts are metadata, not retrievable clause text
            continue
        buffer.append(line)

    flush()
    return chunks


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    """Load every .md file in corpus_dir into a flat list of Chunks."""
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(load_document(path))
    return chunks
