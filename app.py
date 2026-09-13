"""Streamlit chat + compliance dashboard for the Regulatory AI Agent.

Run with:  streamlit run app.py

Uses the offline FakeLLMClient by default so the app is fully demoable
without an API key. Tick "Use live Claude API" in the sidebar (requires
ANTHROPIC_API_KEY in the environment) for real model reasoning.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.regulatory_agent.agent_graph import run_agent
from src.regulatory_agent.llm_client import build_default_client
from src.regulatory_agent.vector_store import RegulatoryVectorStore

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = PROJECT_ROOT / "data" / "regulatory_docs"
PERSIST_DIR = PROJECT_ROOT / "data" / "chroma_store"

st.set_page_config(page_title="Regulatory AI Agent", page_icon="⚖️", layout="wide")


@st.cache_resource
def get_store() -> RegulatoryVectorStore:
    store = RegulatoryVectorStore(persist_dir=PERSIST_DIR)
    if store.size == 0:
        store.index_corpus(CORPUS_DIR)
    return store


store = get_store()

st.title("⚖️ Regulatory AI Agent — Compliance Research Assistant")
st.caption(
    "Demo corpus: FCA SYSC/PRIN, FCA Consumer Duty, Basel III, MiFID II, "
    "UK AML (POCA 2002 / MLR 2017) — paraphrased summaries, not official text."
)

with st.sidebar:
    st.header("Settings")
    live = st.checkbox("Use live Claude API", value=False)
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if live and not has_key:
        st.warning("ANTHROPIC_API_KEY not set — falling back to the offline mock client.")
        live = False
    top_k = st.slider("Clauses to retrieve", 1, 10, 5)
    st.divider()
    st.subheader("Indexed documents")
    for doc in store.list_documents():
        st.write(f"- {doc}")

tab_chat, tab_gap = st.tabs(["Regulatory Q&A", "Compliance Gap Analysis"])

with tab_chat:
    query = st.text_input("Ask a regulatory question", placeholder="e.g. What triggers enhanced due diligence?")
    if st.button("Ask", key="ask_btn") and query:
        llm = build_default_client(mock=not live)
        with st.spinner("Retrieving clauses and reasoning..."):
            result = run_agent(store, llm, query=query, top_k=top_k)
        st.subheader("Answer")
        st.write(result.get("answer"))
        st.subheader("Full briefing")
        st.markdown(result.get("report"))

with tab_gap:
    st.write("Paste an internal policy excerpt to check it against the regulatory corpus.")
    topic = st.text_input("Topic / regulatory area", placeholder="e.g. PEP due diligence")
    policy_text = st.text_area("Internal policy excerpt", height=200)
    if st.button("Analyze gap", key="gap_btn") and topic and policy_text:
        llm = build_default_client(mock=not live)
        with st.spinner("Comparing policy against retrieved clauses..."):
            result = run_agent(store, llm, query=topic, policy_text=policy_text, top_k=top_k)
        st.subheader("Gap analysis")
        st.write(result.get("gap_analysis", {}).get("gap_analysis"))
        st.subheader("Full briefing")
        st.markdown(result.get("report"))
