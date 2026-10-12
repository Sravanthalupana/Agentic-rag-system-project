from collections import deque
from pathlib import Path

import pytest

from rag.agent import MAX_SOURCES, UNKNOWN, RAGAgent, main
from rag.config import Settings
from rag.llm import Decision
from rag.models import Chunk
from rag.store import VectorStore
from tests.fixtures import FakeEmbedder


class ScriptedChat:
    def __init__(self, decisions: list[Decision]) -> None:
        self.decisions = deque(decisions)
        self.payloads: list[dict[str, object]] = []
        self.systems: list[str] = []

    def complete(self, system: str, payload: dict[str, object]) -> Decision:
        self.payloads.append(payload)
        self.systems.append(system)
        return self.decisions.popleft()


def build(
    tmp_path: Path, decisions: list[Decision], settings: Settings | None = None
) -> tuple[RAGAgent, ScriptedChat, FakeEmbedder]:
    embedder, chat = FakeEmbedder(), ScriptedChat(decisions)
    store = VectorStore(tmp_path, embedder.identity)
    store.replace([Chunk("a", "handbook.txt", "lines 1-2", "Deadline: 30 days.")], [[1.0, 0.0]])
    store.replace([Chunk("b", "product.txt", "line 3", "beta refund: 14 days.")], [[0.0, 1.0]])
    return RAGAgent(store, embedder, chat, settings), chat, embedder


def retrieve(*queries: str) -> Decision:
    return Decision(action="retrieve", queries=list(queries))


def test_retrieves_then_answers_with_resolved_citations(tmp_path: Path) -> None:
    agent, chat, _ = build(
        tmp_path,
        [retrieve("deadline"), Decision(action="answer", answer="30 days", citations=["a", "a"])],
    )
    result = agent.ask("When?")
    assert result.answer == "30 days [1]"
    assert [hit.chunk.source for hit in result.citations] == ["handbook.txt"]
    assert result.retrieval_count == 1 and not result.abstained
    assert [step.action for step in result.trace] == ["decide", "retrieve", "answer"]
    assert chat.payloads[1]["sources"][0]["id"] == "a"  # type: ignore[index]
    assert result.latency_ms >= result.llm_ms >= 0 and result.retrieval_ms >= 0


def test_skips_retrieval_for_small_talk(tmp_path: Path) -> None:
    agent, _, embedder = build(
        tmp_path, [Decision(action="answer", answer="Hello!", document_question=False)]
    )
    result = agent.ask("Hello")
    assert result.answer == "Hello!" and result.retrieval_count == 0
    assert not result.citations and not result.abstained and embedder.calls == 0


@pytest.mark.parametrize(
    "decision,reason",
    [
        (Decision(action="answer", answer="30 days", citations=["fake"]), "not retrieved"),
        (Decision(action="answer", answer="30 days"), "no supporting"),
        (Decision(action="answer", answer="30 days", document_question=False), "no supporting"),
        (Decision(action="answer", answer=" ", citations=["a"]), "did not return"),
    ],
)
def test_guards_reject_unsupported_answers(tmp_path: Path, decision: Decision, reason: str) -> None:
    agent, _, _ = build(tmp_path, [retrieve("deadline"), decision])
    result = agent.ask("Deadline?")
    assert result.abstained and result.answer == UNKNOWN and not result.citations
    assert any(step.action == "guard" and reason in step.detail for step in result.trace)


def test_rejects_document_answer_without_retrieval(tmp_path: Path) -> None:
    agent, _, _ = build(tmp_path, [Decision(action="answer", answer="Invented", citations=["a"])])
    assert agent.ask("Deadline?").abstained


def test_abstention_drops_citations(tmp_path: Path) -> None:
    agent, _, _ = build(
        tmp_path,
        [
            retrieve("ceo"),
            Decision(action="answer", answer="No idea", citations=["a"], abstain=True),
        ],
    )
    result = agent.ask("Who is the CEO?")
    assert result.abstained and result.answer == UNKNOWN and not result.citations


def test_multi_query_rounds_and_history(tmp_path: Path) -> None:
    agent, chat, _ = build(
        tmp_path,
        [
            retrieve("reimbursement deadline", "beta refund deadline"),
            retrieve("refund policy"),
            Decision(action="answer", answer="30 and 14 days", citations=["a", "b"]),
        ],
    )
    history = [{"role": "user", "content": "Policies?"}, {"role": "system", "content": "x"}]
    result = agent.ask("Compare them", history)
    assert result.retrieval_count == 3 and result.answer.endswith("[1] [2]")
    assert {hit.chunk.id for hit in result.citations} == {"a", "b"}
    assert chat.payloads[0]["history"][1]["role"] == "user"  # type: ignore[index]
    assert [p["retrievals_left"] for p in chat.payloads] == [3, 2, 1]
    assert "UNTRUSTED DATA" in chat.systems[0]


@pytest.mark.parametrize("rounds", [[["same"], ["SAME "]], [["a"], ["b"], ["c"], ["d"]], [[" "]]])
def test_retrieval_is_bounded_and_deduplicated(tmp_path: Path, rounds: list[list[str]]) -> None:
    agent, _, _ = build(tmp_path, [retrieve(*queries) for queries in rounds])
    result = agent.ask("test?")
    assert result.abstained and result.retrieval_count <= 3
    assert result.trace[-2].action == "guard"


def test_sources_sent_to_model_are_capped(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    store = VectorStore(tmp_path, embedder.identity)
    for i in range(MAX_SOURCES + 6):
        store.replace([Chunk(f"c{i}", f"s{i}.txt", "line 1", "x")], [[1.0, i / 100]])
    chat = ScriptedChat(
        [retrieve("q1"), retrieve("q2"), Decision(action="answer", answer="x", abstain=True)]
    )
    settings = Settings(top_k=MAX_SOURCES + 6)
    RAGAgent(store, embedder, chat, settings).ask("q?")
    assert len(chat.payloads[-1]["sources"]) == MAX_SOURCES  # type: ignore[arg-type]


def test_baseline_retrieves_once_with_raw_question(tmp_path: Path) -> None:
    agent, chat, _ = build(tmp_path, [Decision(action="answer", answer="30 days", citations=["a"])])
    result = agent.baseline("What is the deadline?")
    assert result.retrieval_count == 1 and result.trace[0].detail == "What is the deadline?"
    assert len(chat.payloads) == 1 and chat.payloads[0]["retrievals_left"] == 0
    assert "Single-shot" in chat.systems[0] and result.answer == "30 days [1]"


def test_empty_corpus_and_invalid_questions(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    chat = ScriptedChat([retrieve("deadline"), Decision(action="answer", answer="", abstain=True)])
    agent = RAGAgent(VectorStore(tmp_path, embedder.identity), embedder, chat)
    assert agent.ask("Deadline?").abstained and embedder.calls == 0
    for invalid in ("", "   ", "x" * 2001):
        with pytest.raises(ValueError, match="Ask a nonempty"):
            agent.ask(invalid)
        with pytest.raises(ValueError, match="Ask a nonempty"):
            agent.baseline(invalid)


def test_rejects_invalid_configuration(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    store = VectorStore(tmp_path, embedder.identity)
    with pytest.raises(ValueError, match="retrieval rounds"):
        RAGAgent(store, embedder, ScriptedChat([]), Settings(max_retrievals=9))
    with pytest.raises(ValueError, match="same embedding"):
        RAGAgent(VectorStore(tmp_path / "x", "other"), embedder, ScriptedChat([]))


def test_cli_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rag.ingestion import main as ingest_main

    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    (tmp_path / "doc.txt").write_text("The refund window is 14 days.")
    ingest_main(["--offline", str(tmp_path / "doc.txt")])
    main(["--offline", "How long is the refund window?"])
    output = capsys.readouterr().out
    assert "14 days. [1]" in output and "doc.txt (line 1)" in output and "Latency" in output
