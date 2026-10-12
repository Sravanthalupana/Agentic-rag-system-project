import sqlite3
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from rag.models import Chunk, Hit


def normalized(vectors: list[list[float]]) -> NDArray[np.float32]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Embedding vectors must be rectangular and nonempty.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding vectors must be finite.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Embedding vectors must be nonzero.")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class VectorStore:
    """Local persistent vector store.

    SQLite durably holds chunks and their normalized vectors (atomic per-source replacement);
    an in-memory FAISS inner-product index over them gives cosine search. The index is rebuilt
    lazily whenever the stored revision changes, including writes from other processes.
    """

    def __init__(self, directory: Path, embedding_identity: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "vectors.sqlite3"
        self.identity = embedding_identity
        self._revision = -1
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[Chunk] = []
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, source TEXT NOT NULL, "
                "location TEXT NOT NULL, text TEXT NOT NULL, vector BLOB NOT NULL)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source)")
            existing = connection.execute("SELECT value FROM meta WHERE key='embedding'").fetchone()
            if existing and existing[0] != embedding_identity:
                raise ValueError("Embedding model changed; choose a separate data directory.")
            connection.execute(
                "INSERT OR IGNORE INTO meta VALUES ('embedding', ?)", (embedding_identity,)
            )
            connection.execute("INSERT OR IGNORE INTO meta VALUES ('revision', '0')")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def _bump(connection: sqlite3.Connection) -> None:
        connection.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key='revision'"
        )

    @property
    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT count(*) FROM chunks").fetchone()[0])

    def sources(self) -> dict[str, int]:
        """Indexed source names mapped to their chunk counts."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source, count(*) FROM chunks GROUP BY source ORDER BY source"
            )
            return {str(source): int(count) for source, count in rows}

    def contains(self, chunks: list[Chunk]) -> bool:
        """True if the source is already stored with exactly these chunk IDs."""
        if not chunks:
            return False
        with self._connect() as connection:
            existing = {
                row[0]
                for row in connection.execute(
                    "SELECT id FROM chunks WHERE source=?", (chunks[0].source,)
                )
            }
        return existing == {chunk.id for chunk in chunks}

    def replace(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Atomically replace every chunk of one source."""
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("One embedding is required per chunk.")
        if len({chunk.source for chunk in chunks}) != 1:
            raise ValueError("Replace one source at a time.")
        matrix = normalized(vectors)
        with self._connect() as connection:
            dimension = connection.execute(
                "SELECT value FROM meta WHERE key='dimension'"
            ).fetchone()
            if dimension and int(dimension[0]) != matrix.shape[1]:
                raise ValueError("Embedding dimension mismatch.")
            connection.execute(
                "INSERT OR IGNORE INTO meta VALUES ('dimension', ?)", (str(matrix.shape[1]),)
            )
            connection.execute("DELETE FROM chunks WHERE source=?", (chunks[0].source,))
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [
                    (chunk.id, chunk.source, chunk.location, chunk.text, vector.tobytes())
                    for chunk, vector in zip(chunks, matrix, strict=True)
                ],
            )
            self._bump(connection)

    def delete(self, source: str) -> int:
        with self._connect() as connection:
            removed = connection.execute("DELETE FROM chunks WHERE source=?", (source,)).rowcount
            if removed:
                self._bump(connection)
        return int(removed)

    def _load(self) -> None:
        with self._connect() as connection:
            revision = int(
                connection.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0]
            )
            if revision == self._revision:
                return
            rows = connection.execute(
                "SELECT id, source, location, text, vector FROM chunks ORDER BY id"
            ).fetchall()
        self._chunks = [Chunk(*row[:4]) for row in rows]
        self._index = None
        if rows:
            matrix = np.stack([np.frombuffer(row[4], dtype=np.float32) for row in rows])
            self._index = faiss.IndexFlatIP(matrix.shape[1])
            self._index.add(matrix)
        self._revision = revision

    def search(self, vector: list[float], top_k: int = 4) -> list[Hit]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        self._load()
        if self._index is None:
            return []
        query = normalized([vector])
        if query.shape[1] != self._index.d:
            raise ValueError("Query embedding dimension mismatch.")
        scores, indices = self._index.search(query, min(top_k, len(self._chunks)))
        return [
            Hit(self._chunks[int(i)], float(score))
            for score, i in zip(scores[0], indices[0], strict=True)
            if i >= 0
        ]
