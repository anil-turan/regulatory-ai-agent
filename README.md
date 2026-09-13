# Regulatory AI Agent

A multi-tool RAG agent that answers financial-services compliance questions and
flags gaps between an internal policy and a curated regulatory corpus —
built with **LangGraph**, **ChromaDB**, **sentence-transformers**, and the
**Claude API**.

> **Status**: architecture-complete, fully tested offline. Retrieval and the
> agent state machine are real and tested end to end with no API key.
> LLM reasoning (gap analysis + answer synthesis) is verified with a
> deterministic mock client by default — see [Running live](#running-live-optional)
> to exercise the real Claude API.

## Why this project

Agentic AI over regulatory documents is one of the fastest-growing
intersections in fintech/RegTech (LLM + RAG + domain-specific tools). This
project demonstrates that pattern end to end: retrieval grounded in real
citations, tool-based reasoning rather than a single prompt, and an explicit
LangGraph state machine instead of an implicit agent loop.

## Architecture

```
                     ┌─────────────┐
   query, policy? ──▶│  retrieve   │  regulatory_search (ChromaDB + MiniLM)
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │    route    │  policy_text present? (no LLM call)
                     └──┬───────┬──┘
              cross-ref │       │ gap analysis
                 ┌──────▼──┐ ┌──▼────────────┐
                 │  cross_ │ │ analyze_       │  ← Claude API (or mock)
                 │reference│ │ compliance_gap │
                 └──────┬──┘ └──┬─────────────┘
                        └────┬──┘
                       ┌─────▼─────┐
                       │ synthesize│  ← Claude API (or mock)
                       └─────┬─────┘
                       ┌─────▼─────┐
                       │  report   │  Markdown compliance briefing
                       └───────────┘
```

`route` is a plain conditional edge — no LLM call needed to decide it, since
the caller already tells us whether this is a gap-analysis request (a policy
excerpt was supplied) or a general research question. The LLM is only
invoked where genuine language understanding is required: comparing two
texts for gaps, and synthesising a cited answer from retrieved clauses.

## Tools

| Tool | Needs LLM? | Purpose |
|---|---|---|
| `regulatory_search` | No | Semantic retrieval of the top-k relevant clauses |
| `cross_reference_check` | No | Groups retrieved clauses by source document to reveal overlap across regimes (e.g. AML due diligence in both FCA SYSC and MLR 2017) |
| `analyze_compliance_gap` | Yes | Compares a supplied internal policy excerpt against retrieved clauses and flags gaps, with citations |
| `generate_report` | No | Formats everything into a single Markdown briefing |

## The corpus

`data/regulatory_docs/` contains five documents, chunked at clause level:

- FCA Handbook — SYSC (systems, controls, financial crime, whistleblowing)
- FCA Consumer Duty (PRIN 2A)
- Basel III capital adequacy (CET1, buffers, leverage, liquidity, RWA)
- MiFID II (best execution, product governance, inducements, transaction reporting)
- UK AML framework (POCA 2002 Part 7, MLR 2017)

**These are paraphrased summaries written for this demo, not verbatim
regulatory text.** Every file carries a source note to that effect. Do not
use this corpus or its outputs for real compliance decisions — always check
the official Handbook / legislation.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the vector store (downloads all-MiniLM-L6-v2 once, ~90MB)
python -m src.regulatory_agent.cli --index

# Ask a question (offline mock LLM by default — no API key needed)
python -m src.regulatory_agent.cli --query "What triggers enhanced due diligence?"

# Compliance gap analysis against a policy excerpt
echo "We review broker execution quality once every three years." > /tmp/policy.txt
python -m src.regulatory_agent.cli --query "best execution monitoring" --policy /tmp/policy.txt

# Streamlit dashboard (Q&A tab + gap-analysis tab)
streamlit run app.py
```

## Running live (optional)

Everything above uses `FakeLLMClient`, a deterministic offline stand-in — no
network call, no cost, fully reproducible in CI. To exercise the real Claude
API:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.regulatory_agent.cli --query "..." --live
```

or tick "Use live Claude API" in the Streamlit sidebar.

## Tests

```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

21/21 tests passing, 82% line coverage. The uncovered lines are CLI argument
parsing and the `AnthropicLLMClient` real-API code path (untestable without
a live key by design — see `FakeLLMClient` for how the LLM boundary is
mocked instead).

## Known limitations

- **Retrieval isn't perfect on a small corpus.** `all-MiniLM-L6-v2` is a
  general-purpose embedding model, not fine-tuned for regulatory language.
  On ambiguous queries (e.g. "customer due diligence" overlapping with
  "Consumer Duty"), the top-k results sometimes include a tangentially
  related clause alongside the clearly relevant ones — visible in the
  example output above. A production version would rerank with a
  cross-encoder or fine-tune embeddings on regulatory text.
- **The corpus is curated, not comprehensive.** Five documents, ~48 clauses.
  This is a demonstration of the architecture, not a production regulatory
  database.
- **Gap analysis quality depends on the underlying LLM.** The mock client
  used for testing returns a deterministic placeholder, not real legal
  reasoning — only a live run with `--live` produces genuine gap analysis.

## License

MIT — see `LICENSE`.
