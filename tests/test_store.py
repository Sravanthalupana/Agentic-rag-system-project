from pathlib import Path

import pytest

from rag.models import Chunk
from rag.store import VectorStore


def test_cosine_search_and_persistence(tmp_path: Path) -> None:
    store = VectorStore(tmp_path, "test")
    store.replace([Chunk("a", "a.txt", "text", "alpha")], [[8.0, 0.0]])
    store.replace([Chunk("b", "b.txt", "text", "beta")], [[0.0, 3.0]])
    hits = VectorStore(tmp_path, "test").search([2.0, 0.1], 10)
    assert [hit.chunk.id for hit in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(0.9988, abs=1e-3)
    assert store.sources() == {"a.txt": 1, "b.txt": 1}


def test_index_refreshes_after_writes_from_other_instances(tmp_path: Path) -> None:
    reader = VectorStore(tmp_path, "test")
    writer = VectorStore(tmp_path, "test")
    writer.replace([Chunk("a", "a.txt", "text", "alpha")], [[1.0, 0.0]])
    assert reader.search([1.0, 0.0])[0].chunk.id == "a"
    writer.replace([Chunk("a2", "a.txt", "text", "alpha v2")], [[1.0, 0.0]])
    assert [hit.chunk.id for hit in reader.search([1.0, 0.0])] == ["a2"]
    assert writer.delete("a.txt") == 1
    assert reader.search([1.0, 0.0]) == [] and reader.count == 0


def test_contains_detects_identical_source(tmp_path: Path) -> None:
    store = VectorStore(tmp_path, "test")
    chunks = [Chunk("a", "a.txt", "text", "alpha")]
    assert not store.contains(chunks) and not store.contains([])
    store.replace(chunks, [[1.0, 0.0]])
    assert store.contains(chunks)
    assert not store.contains([Chunk("b", "a.txt", "text", "changed")])


def test_invalid_writes_do_not_destroy_existing_data(tmp_path: Path) -> None:
    store = VectorStore(tmp_path, "test")
    chunk = Chunk("a", "a.txt", "text", "safe")
    store.replace([chunk], [[1.0, 0.0]])
    for vectors in ([], [[1.0]], [[float("nan"), 1.0]], [[0.0, 0.0]]):
        with pytest.raises(ValueError):
            store.replace([chunk], vectors)
        assert store.count == 1
    with pytest.raises(ValueError, match="one source"):
        store.replace([chunk, Chunk("b", "b.txt", "text", "x")], [[1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="model changed"):
        VectorStore(tmp_path, "other")
    with pytest.raises(ValueError, match="dimension"):
        store.search([1.0])
    with pytest.raises(ValueError, match="positive"):
        store.search([1.0, 0.0], 0)


def test_empty_store(tmp_path: Path) -> None:
    store = VectorStore(tmp_path, "test")
    assert store.search([1.0]) == [] and store.count == 0 and store.sources() == {}
