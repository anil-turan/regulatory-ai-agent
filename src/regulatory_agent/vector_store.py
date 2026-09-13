"""RAG retrieval backend: sentence-transformers embeddings + ChromaDB.

Deliberately uses a local embedding model (no API key, no network call at
query time beyond the one-off model download) so the retrieval half of this
project is fully testable without an ANTHROPIC_API_KEY.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils import embedding_functions

from .document_loader import Chunk, load_corpus

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "regulatory_clauses"


class RegulatoryVectorStore:
    """Thin wrapper around a persistent Chroma collection of regulatory clauses."""

    def __init__(self, persist_dir: Path, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        self._persist_dir = persist_dir
        self._embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self._embedder
        )

    @property
    def size(self) -> int:
        return self._collection.count()

    def index_corpus(self, corpus_dir: Path, rebuild: bool = False) -> int:
        """Chunk every document in corpus_dir and upsert into the collection.

        Returns the number of chunks indexed.
        """
        if rebuild:
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME, embedding_function=self._embedder
            )

        chunks = load_corpus(corpus_dir)
        if not chunks:
            return 0

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "doc_title": c.doc_title,
                    "section": c.section,
                    "clause": c.clause,
                    "citation": c.citation(),
                }
                for c in chunks
            ],
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 5, doc_id_filter: str | None = None) -> list[dict]:
        """Return the top_k most relevant clauses for `query`.

        Each result dict has: text, citation, doc_id, section, clause, distance.
        """
        where = {"doc_id": doc_id_filter} if doc_id_filter else None
        result = self._collection.query(query_texts=[query], n_results=top_k, where=where)

        hits: list[dict] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            hits.append({"text": text, "distance": dist, **meta})
        return hits

    def list_documents(self) -> Iterable[str]:
        if self.size == 0:
            return []
        all_meta = self._collection.get()["metadatas"]
        return sorted({m["doc_title"] for m in all_meta})
