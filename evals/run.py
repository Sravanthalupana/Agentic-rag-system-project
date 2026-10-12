"""Score the agentic system against naive single-shot RAG on a fixed question set."""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, quantiles
from tempfile import TemporaryDirectory
from typing import NotRequired, TypedDict

from rag.agent import AgentResult, RAGAgent
from rag.config import Settings
from rag.demo import DemoChat, DemoEmbedder
from rag.embeddings import Embedder, OpenAIEmbedder
from rag.ingestion import ingest
from rag.llm import ChatModel, OpenAIChat
from rag.store import VectorStore

ROOT = Path(__file__).parent
SYSTEMS = ("agentic", "baseline")


class Case(TypedDict):
    id: str
    question: str
    keywords: list[list[str]]
    sources: list[str]
    history: NotRequired[list[dict[str, str]]]
    abstain: NotRequired[bool]


@dataclass(frozen=True)
class Score:
    system: str
    case_id: str
    correct: float
    citation_validity: float
    source_recall: float
    latency_ms: float
    retrievals: int
    answer: str


def load_cases(path: Path = ROOT / "questions.json") -> list[Case]:
    cases: list[Case] = json.loads(path.read_text())
    return cases


def score(system: str, case: Case, result: AgentResult) -> Score:
    """correct: right abstention behaviour and every keyword group present in the answer.
    citation_validity: share of citations that were retrieved in this run.
    source_recall: share of expected source files cited (no citations expected -> none given).
    """
    answer = result.answer.casefold()
    has_keywords = all(
        any(option.casefold() in answer for option in group) for group in case["keywords"]
    )
    correct = float(result.abstained == case.get("abstain", False) and has_keywords)
    retrieved = {
        chunk_id
        for step in result.trace
        if step.action == "retrieve"
        for chunk_id in step.chunk_ids
    }
    if result.citations:
        valid = sum(hit.chunk.id in retrieved for hit in result.citations)
        validity = valid / len(result.citations)
    else:
        validity = float(not case["sources"])
    cited = {hit.chunk.source for hit in result.citations}
    expected = set(case["sources"])
    recall = len(cited & expected) / len(expected) if expected else float(not cited)
    return Score(
        system,
        case["id"],
        correct,
        validity,
        recall,
        result.latency_ms,
        result.retrieval_count,
        result.answer,
    )


def evaluate(agent: RAGAgent, cases: list[Case]) -> list[Score]:
    rows: list[Score] = []
    for i, case in enumerate(cases):
        runs: list[tuple[str, Callable[..., AgentResult]]] = [
            ("agentic", agent.ask),
            ("baseline", agent.baseline),
        ]
        # Alternate order so warm caches or rate limits do not favour one system.
        for name, run in runs if i % 2 == 0 else runs[::-1]:
            rows.append(score(name, case, run(case["question"], case.get("history", []))))
    return rows


def p95(values: list[float]) -> float:
    return quantiles(values, n=20, method="inclusive")[-1] if len(values) > 1 else values[0]


def comparison_table(rows: list[Score]) -> str:
    header = (
        f"{'System':<9} | {'Accuracy':>8} | {'Citation validity':>17} | {'Source recall':>13} "
        f"| {'Mean latency ms':>15} | {'P95 latency ms':>14} | {'Retrievals':>10}"
    )
    lines = [header, "-" * len(header)]
    for name in SYSTEMS:
        group = [row for row in rows if row.system == name]
        latencies = [row.latency_ms for row in group]
        lines.append(
            f"{name:<9} | {mean(r.correct for r in group):>8.1%} | "
            f"{mean(r.citation_validity for r in group):>17.1%} | "
            f"{mean(r.source_recall for r in group):>13.1%} | {mean(latencies):>15.1f} | "
            f"{p95(latencies):>14.1f} | {mean(r.retrievals for r in group):>10.2f}"
        )
    return "\n".join(lines)


def per_question_table(rows: list[Score]) -> str:
    by_case: dict[str, dict[str, Score]] = {}
    for row in rows:
        by_case.setdefault(row.case_id, {})[row.system] = row
    lines = [f"{'Question':<18} | {'agentic':^9} | {'baseline':^9}", "-" * 42]
    for case_id, systems in by_case.items():
        marks = ["pass" if systems[name].correct else "FAIL" for name in SYSTEMS]
        lines.append(f"{case_id:<18} | {marks[0]:^9} | {marks[1]:^9}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Fake providers: harness only")
    parser.add_argument(
        "--chunk-tokens", type=int, default=60, help="Small chunks stress retrieval"
    )
    parser.add_argument("--overlap-tokens", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=2, help="Same for both systems")
    parser.add_argument("--output", type=Path, help="Write per-question JSON results here")
    args = parser.parse_args(argv)
    env = Settings.from_env()
    settings = Settings(
        chat_model=env.chat_model,
        embedding_model=env.embedding_model,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        top_k=args.top_k,
        max_retrievals=env.max_retrievals,
    )
    embedder: Embedder = DemoEmbedder() if args.offline else OpenAIEmbedder(settings)
    model: ChatModel = DemoChat() if args.offline else OpenAIChat(settings)
    cases = load_cases()
    with TemporaryDirectory(prefix="rag-eval-") as directory:
        store = VectorStore(Path(directory), embedder.identity)
        for path in sorted((ROOT / "corpus").glob("*.txt")):
            ingest(path.name, path.read_bytes(), store, embedder, settings)
        rows = evaluate(RAGAgent(store, embedder, model, settings), cases)
        chunks = store.count
    mode = "OFFLINE (deterministic fakes; tests the harness, not model quality)"
    if not args.offline:
        mode = f"OpenAI ({settings.chat_model} + {settings.embedding_model})"
    print(f"Mode: {mode}")
    print(
        f"{len(cases)} fixed questions | {chunks} chunks | top_k={settings.top_k} | "
        f"agent budget={settings.max_retrievals} rounds\n"
    )
    print(comparison_table(rows))
    print()
    print(per_question_table(rows))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mode": mode, "results": [asdict(row) for row in rows]}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
