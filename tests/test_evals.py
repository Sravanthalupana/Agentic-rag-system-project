from pathlib import Path

import pytest

from evals.run import ROOT, Case, comparison_table, evaluate, load_cases, main, p95, score
from rag.agent import AgentResult, RAGAgent, TraceStep
from rag.config import Settings
from rag.demo import DemoChat, DemoEmbedder
from rag.ingestion import ingest
from rag.models import Chunk, Hit
from rag.store import VectorStore


def fact_case() -> Case:
    return {"id": "fact", "question": "Deadline?", "keywords": [["30 days"]], "sources": ["h.txt"]}


def result(
    answer: str, cited: list[Chunk], retrieved: list[str], abstained: bool = False
) -> AgentResult:
    trace = [TraceStep("retrieve", "q", retrieved)] if retrieved else []
    return AgentResult(answer, [Hit(c, 1.0) for c in cited], trace, 5.0, abstained)


def test_question_set_is_well_formed() -> None:
    cases = load_cases()
    assert len({case["id"] for case in cases}) == len(cases) >= 10
    corpus = {path.name for path in (ROOT / "corpus").glob("*.txt")}
    assert all(set(case["sources"]) <= corpus for case in cases)
    assert any(case.get("abstain") for case in cases)


def test_score_correct_answer() -> None:
    row = score(
        "agentic", fact_case(), result("It is 30 days [1]", [Chunk("a", "h.txt", "l", "")], ["a"])
    )
    assert (row.correct, row.citation_validity, row.source_recall) == (1, 1, 1)


def test_score_wrong_answer_and_invalid_citation() -> None:
    row = score("agentic", fact_case(), result("99 days", [Chunk("x", "o.txt", "l", "")], ["a"]))
    assert row.correct == row.citation_validity == row.source_recall == 0


def test_score_abstention_cases() -> None:
    unknown: Case = {"id": "u", "question": "CEO?", "keywords": [], "sources": [], "abstain": True}
    assert score("a", unknown, result("Unknown", [], [], abstained=True)).correct == 1
    assert score("a", unknown, result("Bob", [Chunk("a", "h.txt", "l", "")], ["a"])).correct == 0
    assert score("a", fact_case(), result("Unknown", [], ["a"], abstained=True)).correct == 0


def test_p95() -> None:
    assert p95([5.0]) == 5.0
    assert p95([float(i) for i in range(1, 101)]) == pytest.approx(95.05)


def test_offline_comparison_is_repeatable_and_fair(tmp_path: Path) -> None:
    settings = Settings(chunk_tokens=60, overlap_tokens=10, top_k=2)
    embedder = DemoEmbedder()
    store = VectorStore(tmp_path, embedder.identity)
    for path in sorted((ROOT / "corpus").glob("*.txt")):
        ingest(path.name, path.read_bytes(), store, embedder, settings)
    agent = RAGAgent(store, embedder, DemoChat(), settings)
    cases = load_cases()
    first, second = evaluate(agent, cases), evaluate(agent, cases)
    assert len(first) == 2 * len(cases)
    assert [(r.system, r.correct, r.answer) for r in first] == [
        (r.system, r.correct, r.answer) for r in second
    ]
    assert all(r.retrievals == 1 for r in first if r.system == "baseline")
    assert any(r.retrievals == 0 for r in first if r.system == "agentic")
    assert all(r.citation_validity == 1 for r in first)
    table = comparison_table(first)
    assert "agentic" in table and "baseline" in table and "Citation validity" in table


def test_cli_offline_prints_table_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out" / "eval.json"
    main(["--offline", "--output", str(output)])
    printed = capsys.readouterr().out
    assert "OFFLINE" in printed and "Accuracy" in printed and "compare-deadlines" in printed
    assert '"system": "baseline"' in output.read_text()
