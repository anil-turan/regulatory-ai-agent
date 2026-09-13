from pathlib import Path

from src.regulatory_agent.document_loader import load_corpus, load_document

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "regulatory_docs"


def test_load_document_parses_clauses():
    chunks = load_document(CORPUS_DIR / "fca_sysc_principles.md")
    assert len(chunks) > 5
    assert all(c.text.strip() for c in chunks)
    assert all(c.clause for c in chunks)


def test_chunk_citation_format():
    chunks = load_document(CORPUS_DIR / "basel_iii_capital.md")
    citation = chunks[0].citation()
    assert " — " in citation
    assert chunks[0].doc_title in citation


def test_source_note_callouts_excluded_from_chunk_text():
    chunks = load_document(CORPUS_DIR / "aml_poca_regs.md")
    for c in chunks:
        assert "Source note" not in c.text


def test_load_corpus_covers_all_five_documents():
    chunks = load_corpus(CORPUS_DIR)
    doc_ids = {c.doc_id for c in chunks}
    assert doc_ids == {
        "fca_sysc_principles",
        "basel_iii_capital",
        "mifid_ii_best_execution",
        "aml_poca_regs",
        "fca_consumer_duty",
    }


def test_chunk_ids_are_unique():
    chunks = load_corpus(CORPUS_DIR)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
