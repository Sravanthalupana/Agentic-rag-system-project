from typing import Protocol

from openai import OpenAI

from rag.config import Settings, api_key

BATCH_SIZE = 128


class Embedder(Protocol):
    @property
    def identity(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.embedding_model
        self.client = OpenAI(api_key=api_key(), timeout=45.0, max_retries=2)

    @property
    def identity(self) -> str:
        return f"openai:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            response = self.client.embeddings.create(
                model=self.model, input=texts[start : start + BATCH_SIZE]
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors
