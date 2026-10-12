"""Deterministic offline providers for demos and offline evals; not OpenAI quality."""

import hashlib
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.llm import Decision

STOP_WORDS = frozenset(
    "a an the what which who how is are was were to of for in on and or i it that this do does "
    "can me my we our you your please tell about much many when where long be by with from".split()
)
DIMENSIONS = 512


def words(text: str) -> list[str]:
    """Lowercased content words with a naive plural strip ("customers" -> "customer")."""
    found = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.casefold())
    return [
        w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        for w in found
        if w not in STOP_WORDS
    ]


def terms(text: str) -> set[str]:
    return set(words(text))


class DemoEmbedder:
    """Hashed bag-of-words vectors: lexical similarity, no network."""

    identity = "offline:hashed-words-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * DIMENSIONS
            for word, count in Counter(words(text)).items():
                bucket = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % DIMENSIONS
                vector[bucket] += count
            if not any(vector):
                vector[0] = 1.0
            vectors.append(vector)
        return vectors


GREETINGS = frozenset({"hi", "hello", "hey", "thanks", "thank you", "good morning"})
PRONOUNS = re.compile(r"\b(it|that|they|those|them|their|this)\b")


def sub_questions(question: str) -> list[str]:
    """Split comparison/compound questions into parts (at most 3)."""
    text = re.sub(r"^\s*(compare|contrast)\s+", "", question.strip(), flags=re.IGNORECASE)
    parts = [part.strip(" ?.") for part in re.split(r"\s+(?:and|vs\.?|versus)\s+", text)]
    parts = [part for part in parts if terms(part)]
    return parts[:3] if len(parts) > 1 else [question.strip()]


class DemoChat:
    """Rule-based stand-in for the chat model: query planning plus extractive answers."""

    def complete(self, system: str, payload: dict[str, object]) -> "Decision":
        from rag.llm import Decision

        question = str(payload["question"]).strip()
        if question.casefold().rstrip("!. ") in GREETINGS:
            return Decision(
                action="answer",
                answer="Hello! Ask me anything about your uploaded documents.",
                document_question=False,
                reason="Small talk; no retrieval needed.",
            )
        history = [turn for turn in _list(payload["history"]) if turn.get("role") == "user"]
        context = history[-1]["content"] if history and PRONOUNS.search(question.lower()) else ""
        parts = sub_questions(question)
        sources = _list(payload["sources"])
        if not sources and int(str(payload["retrievals_left"])) > 0:
            queries = [f"{context} {part}".strip() for part in parts]
            return Decision(action="retrieve", queries=queries, reason="Search for each part.")
        sentences = [
            (sentence.strip(), source["id"])
            for source in sources
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", source["text"])
            if len(sentence.split()) >= 4
        ]
        answer: list[str] = []
        citations: list[str] = []
        for part in parts:
            focus, background = terms(part), terms(context)
            scored = [
                (2 * len(focus & terms(s)) + len(background & terms(s)), s, chunk_id)
                for s, chunk_id in sentences
                if focus & terms(s) and s not in answer
            ]
            if not scored:
                return Decision(action="answer", answer="", abstain=True, reason="No evidence.")
            _, sentence, chunk_id = max(scored, key=lambda item: item[0])
            answer.append(sentence)
            citations.append(chunk_id)
        return Decision(
            action="answer",
            answer=" ".join(dict.fromkeys(answer)),
            citations=list(dict.fromkeys(citations)),
            reason="Extracted matching sentences.",
        )


def _list(value: object) -> list[dict[str, str]]:
    return list(value) if isinstance(value, list) else []
