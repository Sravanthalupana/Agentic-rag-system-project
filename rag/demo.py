"""Deterministic offline providers for demos and offline evals; not OpenAI quality."""

import hashlib
import re
from collections import Counter

STOP_WORDS = frozenset(
    "a an the what which who how is are was were to of for in on and or i it that this do does "
    "can me my we our you your please tell about much many when where long be by with from".split()
)
DIMENSIONS = 512


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.casefold())) - STOP_WORDS


class DemoEmbedder:
    """Hashed bag-of-words vectors: lexical similarity, no network."""

    identity = "offline:hashed-words-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * DIMENSIONS
            words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.casefold())
            for word, count in Counter(w for w in words if w not in STOP_WORDS).items():
                bucket = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % DIMENSIONS
                vector[bucket] += count
            if not any(vector):
                vector[0] = 1.0
            vectors.append(vector)
        return vectors
