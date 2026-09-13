"""Command-line entry point.

    python -m regulatory_agent.cli --index                     # build the vector store
    python -m regulatory_agent.cli --query "PEP due diligence"  # mock run (default)
    python -m regulatory_agent.cli --query "..." --live         # real Claude API call
    python -m regulatory_agent.cli --query "..." --policy policy.txt --live
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .agent_graph import run_agent
from .llm_client import build_default_client
from .vector_store import RegulatoryVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "regulatory_docs"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "data" / "chroma_store"


def main() -> None:
    parser = argparse.ArgumentParser(description="Regulatory AI Agent CLI")
    parser.add_argument("--index", action="store_true", help="(Re)build the vector store from data/regulatory_docs")
    parser.add_argument("--query", type=str, help="Question to ask the agent")
    parser.add_argument("--policy", type=Path, help="Path to a text file with an internal policy excerpt (triggers gap analysis)")
    parser.add_argument("--live", action="store_true", help="Use the real Claude API instead of the offline mock client")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    store = RegulatoryVectorStore(persist_dir=DEFAULT_PERSIST_DIR)

    if args.index or store.size == 0:
        n = store.index_corpus(DEFAULT_CORPUS_DIR, rebuild=args.index)
        print(f"Indexed {n} clauses from {DEFAULT_CORPUS_DIR}" if n else "Vector store already populated.")

    if not args.query:
        print(f"Documents indexed: {list(store.list_documents())}")
        return

    llm = build_default_client(mock=not args.live)
    policy_text = args.policy.read_text(encoding="utf-8") if args.policy else None

    result = run_agent(store, llm, query=args.query, policy_text=policy_text, top_k=args.top_k)

    print("\n=== Answer ===\n")
    print(result.get("answer", "(no answer generated)"))
    print("\n=== Full Report ===\n")
    print(result.get("report", "(no report generated)"))


if __name__ == "__main__":
    main()
