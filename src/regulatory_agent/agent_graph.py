"""LangGraph state machine wiring the four tools into a single agent.

Flow:

    START -> retrieve -> route -> [cross_reference | gap_analysis] -> synthesize -> report -> END

`route` is a plain conditional edge (no LLM call needed to decide it): if the
caller supplied `policy_text`, this is a compliance-gap request, otherwise
it's a general research request and we cross-reference across documents
instead. The LLM is only invoked where genuine language understanding is
required: gap analysis (comparing two texts) and the final synthesis answer.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from .llm_client import LLMClient
from .tools import (
    analyze_compliance_gap,
    cross_reference_check,
    generate_report,
    regulatory_search,
)
from .vector_store import RegulatoryVectorStore

SYNTHESIS_SYSTEM_PROMPT = """\
You are a financial-services regulatory research assistant. Answer the
user's question using ONLY the retrieved clauses provided. Always cite the
clause (document — section — clause) for every claim. If the retrieved
clauses do not answer the question, say so plainly rather than guessing.
"""


class AgentState(TypedDict, total=False):
    query: str
    policy_text: str | None
    top_k: int
    search_hits: list[dict]
    cross_reference: dict
    gap_analysis: dict
    answer: str
    report: str


def build_agent_graph(store: RegulatoryVectorStore, llm: LLMClient):
    def retrieve(state: AgentState) -> AgentState:
        hits = regulatory_search(store, state["query"], top_k=state.get("top_k", 5))
        return {"search_hits": hits}

    def route(state: AgentState) -> Literal["gap_analysis", "cross_reference"]:
        return "gap_analysis" if state.get("policy_text") else "cross_reference"

    def cross_reference_node(state: AgentState) -> AgentState:
        return {"cross_reference": cross_reference_check(store, state["query"])}

    def gap_analysis_node(state: AgentState) -> AgentState:
        result = analyze_compliance_gap(
            store, llm, policy_text=state["policy_text"], topic=state["query"]
        )
        return {"gap_analysis": result}

    def synthesize(state: AgentState) -> AgentState:
        hits = state.get("search_hits", [])
        if not hits:
            prompt = f"Question: {state['query']}\n\nNo relevant clauses were found."
        else:
            clause_block = "\n\n".join(f"- {h['citation']}:\n  {h['text']}" for h in hits)
            prompt = f"Question: {state['query']}\n\nRetrieved clauses:\n{clause_block}"
        answer = llm.complete(system=SYNTHESIS_SYSTEM_PROMPT, prompt=prompt)
        return {"answer": answer}

    def report(state: AgentState) -> AgentState:
        text = generate_report(
            topic=state["query"],
            search_hits=state.get("search_hits", []),
            cross_reference=state.get("cross_reference"),
            gap_analysis=state.get("gap_analysis"),
        )
        return {"report": text}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("cross_reference", cross_reference_node)
    graph.add_node("gap_analysis", gap_analysis_node)
    graph.add_node("synthesize", synthesize)
    graph.add_node("report", report)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve", route, {"gap_analysis": "gap_analysis", "cross_reference": "cross_reference"}
    )
    graph.add_edge("cross_reference", "synthesize")
    graph.add_edge("gap_analysis", "synthesize")
    graph.add_edge("synthesize", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_agent(
    store: RegulatoryVectorStore,
    llm: LLMClient,
    query: str,
    policy_text: str | None = None,
    top_k: int = 5,
) -> AgentState:
    app = build_agent_graph(store, llm)
    return app.invoke({"query": query, "policy_text": policy_text, "top_k": top_k})
