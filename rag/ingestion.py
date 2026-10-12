import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rag.config import Settings
from rag.demo import DemoEmbedder
from rag.documents import SUPPORTED_SUFFIXES, chunk_sections, load_document
from rag.embeddings import Embedder, OpenAIEmbedder
from rag.store import VectorStore


@dataclass(frozen=True)
class IngestResult:
    source: str
    chunks: int
    skipped: bool


def ingest(
    filename: str, data: bytes, store: VectorStore, embedder: Embedder, settings: Settings
) -> IngestResult:
    """Load, chunk, embed, and store one document; unchanged documents are not re-embedded."""
    if store.identity != embedder.identity:
        raise ValueError("Store and embedder must use the same embedding model.")
    chunks = chunk_sections(
        load_document(filename, data), settings.chunk_tokens, settings.overlap_tokens
    )
    if not chunks:
        raise ValueError("No readable text found. Scanned PDFs require OCR before upload.")
    if store.contains(chunks):
        return IngestResult(chunks[0].source, len(chunks), skipped=True)
    vectors = embedder.embed([chunk.text for chunk in chunks])
    store.replace(chunks, vectors)
    return IngestResult(chunks[0].source, len(chunks), skipped=False)


def discover(paths: Iterable[Path]) -> list[Path]:
    """Expand directories recursively into supported document files."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(
                sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
            )
        else:
            found.append(path)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index PDF, DOCX, TXT, or MD files locally.")
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories")
    parser.add_argument(
        "--offline", action="store_true", help="Deterministic offline embeddings (no API key)"
    )
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    embedder: Embedder = DemoEmbedder() if args.offline else OpenAIEmbedder(settings)
    directory = settings.data_dir / "offline" if args.offline else settings.data_dir
    store = VectorStore(directory, embedder.identity)
    failures = 0
    for path in discover(args.paths):
        try:
            result = ingest(path.name, path.read_bytes(), store, embedder, settings)
        except (OSError, ValueError) as error:
            failures += 1
            print(f"{path}: FAILED - {error}")
            continue
        status = "unchanged" if result.skipped else "indexed"
        print(f"{path.name}: {result.chunks} chunks {status}")
    print(f"Store: {store.count} chunks from {len(store.sources())} sources in {directory}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
