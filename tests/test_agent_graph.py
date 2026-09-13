from src.regulatory_agent.agent_graph import run_agent


def test_run_agent_research_mode_uses_cross_reference_branch(indexed_store, fake_llm):
    result = run_agent(indexed_store, fake_llm, query="suspicious activity report obligations")
    assert "search_hits" in result
    assert "cross_reference" in result
    assert "gap_analysis" not in result
    assert result["answer"]
    assert "Regulatory Compliance Briefing" in result["report"]


def test_run_agent_gap_mode_uses_gap_analysis_branch(indexed_store, fake_llm):
    result = run_agent(
        indexed_store,
        fake_llm,
        query="best execution monitoring",
        policy_text="We review broker execution quality once every three years.",
    )
    assert "gap_analysis" in result
    assert "cross_reference" not in result
    assert result["gap_analysis"]["clauses_considered"] > 0


def test_run_agent_no_hits_still_produces_report(empty_store, fake_llm):
    result = run_agent(empty_store, fake_llm, query="something not in the corpus at all")
    assert result["search_hits"] == []
    assert "No relevant clauses were found" in result["report"]


def test_llm_called_at_least_once_per_run(indexed_store, fake_llm):
    run_agent(indexed_store, fake_llm, query="capital conservation buffer")
    assert len(fake_llm.calls) >= 1
