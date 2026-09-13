import shutil
from pathlib import Path

import pytest

from src.regulatory_agent.llm_client import FakeLLMClient
from src.regulatory_agent.vector_store import RegulatoryVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "data" / "regulatory_docs"


@pytest.fixture(scope="session")
def indexed_store(tmp_path_factory):
    persist_dir = tmp_path_factory.mktemp("chroma_test_store")
    store = RegulatoryVectorStore(persist_dir=persist_dir)
    store.index_corpus(CORPUS_DIR)
    yield store
    shutil.rmtree(persist_dir, ignore_errors=True)


@pytest.fixture
def empty_store(tmp_path):
    persist_dir = tmp_path / "empty_chroma_store"
    return RegulatoryVectorStore(persist_dir=persist_dir)


@pytest.fixture
def fake_llm():
    return FakeLLMClient()
