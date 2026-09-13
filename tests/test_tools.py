from src.regulatory_agent.tools import (
    analyze_compliance_gap,
    cross_reference_check,
    generate_report,
    regulatory_search,
)


def test_regulatory_search_empty_store_returns_empty_list(empty_store):
    assert regulatory_search(empty_store, "money laundering") == []


def test_regulatory_search_returns_hits(indexed_store):
    hits = regulatory_search(indexed_store, "best execution factors", top_k=3)
    assert 0 < len(hits) <= 3
    assert all("citation" in h for h in hits)


def test_cross_reference_detects_multi_regime_overlap(indexed_store):
    result = cross_reference_check(indexed_store, "record keeping retention period", top_k=10)
    assert result["topic"] == "record keeping retention period"
    assert result["documents_covered"] >= 1
    assert isinstance(result["by_document"], dict)


def test_gap_analyzer_calls_llm_with_policy_and_clauses(indexed_store, fake_llm):
    policy = "We verify customer identity using a passport or driving licence at onboarding."
    result = analyze_compliance_gap(
        indexed_store, fake_llm, policy_text=policy, topic="customer due diligence"
    )
    assert result["clauses_considered"] > 0
    assert len(fake_llm.calls) == 1
    assert policy in fake_llm.calls[0]["prompt"]
    assert result["gap_analysis"]


def test_gap_analyzer_handles_no_clauses_found(empty_store, fake_llm):
    result = analyze_compliance_gap(
        empty_store, fake_llm, policy_text="Some policy text.", topic="nonexistent topic xyz"
    )
    assert result["clauses_considered"] == 0
    assert "No matching regulatory clauses" in result["gap_analysis"]


def test_generate_report_with_no_hits():
    report = generate_report("some topic", search_hits=[])
    assert "No relevant clauses were found" in report


def test_generate_report_includes_citations_and_cross_reference():
    hits = [{"citation": "Doc A — Sec 1 — Clause 1", "text": "Example clause text."}]
    cross_ref = {
        "multi_regime_overlap": True,
        "documents_covered": 2,
        "by_document": {"Doc A": [{"clause": "Clause 1"}], "Doc B": [{"clause": "Clause 2"}]},
    }
    report = generate_report("topic", search_hits=hits, cross_reference=cross_ref)
    assert "Doc A — Sec 1 — Clause 1" in report
    assert "Cross-Regime Overlap" in report
    assert "Doc B" in report
