"""Thin wrapper around the Claude API.

Kept as a single narrow interface (`LLMClient.complete`) so every test in this
project can substitute a `FakeLLMClient` and exercise the full agent graph
deterministically, with zero network calls and no API key required.

Set ANTHROPIC_API_KEY to use `AnthropicLLMClient` for a real, live run — see
README.md "Running live" section.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        """Return the model's text completion for the given system + user prompt."""
        raise NotImplementedError


class AnthropicLLMClient(LLMClient):
    """Real Claude API client. Requires the `anthropic` package and an API key."""

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None):
        import anthropic  # imported lazily so the package is optional in mock-only runs

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it to run the agent live, "
                "or use FakeLLMClient / --mock for a network-free run."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class FakeLLMClient(LLMClient):
    """Deterministic stand-in for tests and offline demos.

    Rather than returning a canned string regardless of input, it does a
    lightweight extractive summary of the retrieved clauses passed in the
    prompt, so agent-graph tests exercise real conditional logic (e.g. "no
    clauses found" vs "clauses found") without ever calling out to a real LLM.
    """

    def __init__(self, canned_responses: dict[str, str] | None = None):
        self._canned = canned_responses or {}
        self.calls: list[dict] = []

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.calls.append({"system": system, "prompt": prompt, "max_tokens": max_tokens})
        for key, response in self._canned.items():
            if key in prompt:
                return response

        if "No relevant clauses were found" in prompt:
            return (
                "No matching regulatory clauses were retrieved for this query. "
                "Recommend broadening the search terms or confirming the topic "
                "falls within the indexed corpus (FCA SYSC/PRIN, Basel III, "
                "MiFID II, UK AML/POCA, FCA Consumer Duty)."
            )

        return (
            "[FakeLLMClient mock summary] Based on the retrieved clauses above, "
            "the relevant obligations have been identified with citations. "
            "This is a deterministic offline stand-in — run with a real "
            "ANTHROPIC_API_KEY for genuine model reasoning."
        )


def build_default_client(mock: bool = True) -> LLMClient:
    if mock:
        return FakeLLMClient()
    return AnthropicLLMClient()
