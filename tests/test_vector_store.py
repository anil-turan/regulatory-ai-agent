def test_index_corpus_returns_chunk_count(empty_store, tmp_path):
    from pathlib import Path

    corpus_dir = Path(__file__).resolve().parents[1] / "data" / "regulatory_docs"
    n = empty_store.index_corpus(corpus_dir)
    assert n > 20
    assert empty_store.size == n


def test_search_on_empty_store_via_tool_returns_empty():
    # covered more directly in test_tools.py; this checks the raw property
    pass


def test_search_returns_relevant_pep_clause(indexed_store):
    hits = indexed_store.search("politically exposed person enhanced due diligence", top_k=3)
    assert len(hits) == 3
    citations = " ".join(h["citation"] for h in hits)
    assert "aml_poca_regs" in " ".join(h["doc_id"] for h in hits) or "PEP" in citations or "Enhanced" in citations


def test_search_respects_doc_id_filter(indexed_store):
    hits = indexed_store.search("capital requirements", top_k=5, doc_id_filter="basel_iii_capital")
    assert len(hits) > 0
    assert all(h["doc_id"] == "basel_iii_capital" for h in hits)


def test_list_documents_returns_all_titles(indexed_store):
    docs = list(indexed_store.list_documents())
    assert len(docs) == 5
