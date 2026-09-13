"""Tool 1: regulatory_search — semantic retrieval over the indexed corpus."""
from __future__ import annotations

from ..vector_store import RegulatoryVectorStore


def regulatory_search(
    store: RegulatoryVectorStore, query: str, top_k: int = 5, doc_id_filter: str | None = None
) -> list[dict]:
    """Retrieve the top_k clauses most relevant to `query`.

    Returns [] if the store is empty or nothing clears Chroma's default
    similarity threshold behaviour (Chroma itself doesn't threshold, so we
    additionally drop hits with cosine distance > 1.3 — L2-normalised
    embeddings from all-MiniLM-L6-v2 rarely exceed this for genuinely
    unrelated text, based on manual inspection of this project's corpus).
    """
    if store.size == 0:
        return []

    hits = store.search(query, top_k=top_k, doc_id_filter=doc_id_filter)
    return [h for h in hits if h["distance"] <= 1.3]
