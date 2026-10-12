from pathlib import Path

import pytest

from rag.agent import RAGAgent
from rag.config import Settings
from rag.demo import DemoChat, DemoEmbedder, sub_questions
from rag.ingestion import ingest
from rag.store import VectorStore

CORPUS = Path(__file__).parents[1] / "evals" / "corpus"


@pytest.fixture
def agent(tmp_path: Path) -> RAGAgent:
    settings = Settings(chunk_tokens=40, overlap_tokens=5)
    embedder = DemoEmbedder()
    store = VectorStore(tmp_path, embedder.identity)
    for path in sorted(CORPUS.glob("*.txt")):
        ingest(path.name, path.read_bytes(), store, embedder, settings)
    return RAGAgent(store, embedder, DemoChat(), settings)


def test_sub_questions() -> None:
    assert sub_questions("Compare the refund deadline and the warranty?") == [
        "the refund deadline",
        "the warranty",
    ]
    assert sub_questions("What is the deadline?") == ["What is the deadline?"]


def test_offline_agent_answers_with_citations(agent: RAGAgent) -> None:
    result = agent.ask("What is the reimbursement deadline?")
    assert "30 days" in result.answer and result.citations[0].chunk.source == "handbook.txt"


def test_offline_agent_resolves_follow_up(agent: RAGAgent) -> None:
    history = [{"role": "user", "content": "What is the reimbursement deadline?"}]
    assert "expense portal" in agent.ask("How do I submit it?", history).answer


def test_offline_agent_splits_comparisons(agent: RAGAgent) -> None:
    result = agent.ask("Compare the reimbursement deadline and the customer refund deadline.")
    assert result.retrieval_count == 2
    assert "30 days" in result.answer and "14 days" in result.answer
    assert {hit.chunk.source for hit in result.citations} == {"handbook.txt", "product.txt"}


def test_offline_agent_small_talk_and_abstention(agent: RAGAgent) -> None:
    hello = agent.ask("Hello")
    assert hello.retrieval_count == 0 and not hello.abstained
    assert agent.ask("Who is the CEO?").abstained
