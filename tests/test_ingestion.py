from pathlib import Path

import pytest

from rag.config import Settings
from rag.demo import DemoEmbedder
from rag.ingestion import discover, ingest, main
from rag.store import VectorStore
from tests.fixtures import FakeEmbedder, docx_bytes, pdf_bytes


def test_ingest_is_idempotent_and_replaces_changed_source(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    store = VectorStore(tmp_path, embedder.identity)
    first = ingest("a.txt", b"alpha content", store, embedder, Settings())
    assert (first.chunks, first.skipped) == (1, False)
    again = ingest("a.txt", b"alpha content", store, embedder, Settings())
    assert again.skipped and store.count == 1 and embedder.calls == 1
    ingest("a.txt", b"beta content", store, embedder, Settings())
    assert store.count == 1 and embedder.calls == 2
    assert store.search([0.0, 1.0])[0].chunk.text == "beta content"


def test_ingest_all_formats(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    store = VectorStore(tmp_path, embedder.identity)
    ingest("w.pdf", pdf_bytes("Warranty is 2 years."), store, embedder, Settings())
    ingest("s.docx", docx_bytes(), store, embedder, Settings())
    ingest("n.md", b"# Notes\n\nbeta", store, embedder, Settings())
    assert store.sources() == {"n.md": 1, "s.docx": 1, "w.pdf": 1}
    assert store.search([0.0, 1.0], 1)[0].chunk.location == "line 1 – line 3"


def test_rejects_mismatched_embedder(tmp_path: Path) -> None:
    store = VectorStore(tmp_path, "other")
    with pytest.raises(ValueError, match="same embedding model"):
        ingest("a.txt", b"x", store, FakeEmbedder(), Settings())


def test_discover_expands_directories(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    for name in ("a.txt", "sub/b.PDF", "sub/c.docx", "skip.exe"):
        (tmp_path / name).write_bytes(b"x")
    names = [path.name for path in discover([tmp_path])]
    assert names == ["a.txt", "b.PDF", "c.docx"]


def test_cli_offline_indexes_and_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    good, bad = tmp_path / "good.txt", tmp_path / "bad.pdf"
    good.write_text("The refund window is 14 days.")
    bad.write_bytes(b"not a pdf")
    assert main(["--offline", str(good), str(bad)]) == 1
    output = capsys.readouterr().out
    assert "good.txt: 1 chunks indexed" in output and "bad.pdf: FAILED" in output
    store = VectorStore(tmp_path / "data" / "offline", DemoEmbedder.identity)
    assert store.sources() == {"good.txt": 1}
    assert main(["--offline", str(good)]) == 0
    assert "unchanged" in capsys.readouterr().out
