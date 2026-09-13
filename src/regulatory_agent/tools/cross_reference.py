"""Tool 2: cross_reference_check — find the same topic across multiple
regulatory documents, so an analyst can see where obligations overlap or
where one regime is stricter than another (e.g. AML due diligence appears in
both FCA SYSC 6.3 and MLR 2017)."""
from __future__ import annotations

from collections import defaultdict

from ..vector_store import RegulatoryVectorStore
from .regulatory_search import regulatory_search


def cross_reference_check(store: RegulatoryVectorStore, topic: str, top_k: int = 8) -> dict:
    """Group retrieved clauses by source document to reveal cross-regime overlap.

    Returns a dict: {"topic": ..., "documents_covered": int, "by_document": {...}}
    """
    hits = regulatory_search(store, topic, top_k=top_k)

    by_document: dict[str, list[dict]] = defaultdict(list)
    for hit in hits:
        by_document[hit["doc_title"]].append(
            {"clause": hit["clause"], "citation": hit["citation"], "distance": hit["distance"]}
        )

    return {
        "topic": topic,
        "documents_covered": len(by_document),
        "by_document": dict(by_document),
        "multi_regime_overlap": len(by_document) >= 2,
    }
