"""Tool 3: analyze_compliance_gap — compare a supplied internal policy
snippet against the retrieved regulatory clauses and flag gaps.

This is the one tool that genuinely needs an LLM (semantic comparison of two
texts is not a good fit for keyword or embedding-distance heuristics alone),
so it's the clearest place the FakeLLMClient/AnthropicLLMClient seam matters.
"""
from __future__ import annotations

from ..llm_client import LLMClient
from ..vector_store import RegulatoryVectorStore
from .regulatory_search import regulatory_search

GAP_ANALYSIS_SYSTEM_PROMPT = """\
You are a financial-services regulatory compliance analyst. You will be given
an internal policy excerpt and a set of retrieved regulatory clauses relevant
to it. Identify specific gaps: obligations present in the regulatory clauses
that are not clearly addressed by the internal policy. Be precise and cite
the clause (document — section — clause) for every gap you raise. If the
policy fully covers the retrieved clauses, say so explicitly rather than
inventing a gap. Do not comment on anything outside the retrieved clauses.
"""


def analyze_compliance_gap(
    store: RegulatoryVectorStore, llm: LLMClient, policy_text: str, topic: str, top_k: int = 5
) -> dict:
    """Return {"topic", "clauses_considered", "gap_analysis", "citations"}."""
    hits = regulatory_search(store, topic, top_k=top_k)

    if not hits:
        prompt = (
            f"Internal policy excerpt:\n{policy_text}\n\n"
            "No relevant clauses were found in the regulatory corpus for "
            f"topic '{topic}'."
        )
    else:
        clause_block = "\n\n".join(f"- {h['citation']}:\n  {h['text']}" for h in hits)
        prompt = (
            f"Internal policy excerpt:\n{policy_text}\n\n"
            f"Retrieved regulatory clauses for topic '{topic}':\n{clause_block}"
        )

    analysis = llm.complete(system=GAP_ANALYSIS_SYSTEM_PROMPT, prompt=prompt)

    return {
        "topic": topic,
        "clauses_considered": len(hits),
        "gap_analysis": analysis,
        "citations": [h["citation"] for h in hits],
    }
