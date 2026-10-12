import hashlib
from collections.abc import Iterator
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

import tiktoken
from docx import Document
from docx.table import Table
from pypdf import PdfReader

from rag.models import Chunk, Section

SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_TEXT_CHARS = 2_000_000
ENCODING = tiktoken.get_encoding("cl100k_base")


def load_document(filename: str, data: bytes) -> list[Section]:
    """Extract located text sections; raises ValueError with a user-safe message."""
    source = PurePosixPath(filename.replace("\\", "/")).name
    suffix = PurePosixPath(source).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Supported formats: PDF, DOCX, TXT, MD.")
    if not data or len(data) > MAX_FILE_BYTES:
        raise ValueError("Document must be nonempty and at most 20 MB.")
    try:
        if suffix == ".pdf":
            sections = _pdf_sections(source, data)
        elif suffix == ".docx":
            sections = _docx_sections(source, data)
        else:
            sections = _text_sections(source, data.decode("utf-8-sig"))
    except (UnicodeError, BadZipFile) as exc:
        raise ValueError("Unreadable document; text files must use UTF-8.") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Could not read this document; check the file format.") from exc
    sections = [section for section in sections if section.text.strip()]
    if not sections:
        raise ValueError("No readable text found. Scanned PDFs require OCR before upload.")
    if sum(len(section.text) for section in sections) > MAX_TEXT_CHARS:
        raise ValueError("Extracted text exceeds 2 million characters.")
    return sections


def _text_sections(source: str, text: str) -> list[Section]:
    """One section per blank-line separated paragraph, located by line range."""
    sections: list[Section] = []
    lines = text.splitlines()
    start = 0
    for number, line in enumerate([*lines, ""], 1):
        if line.strip():
            continue
        paragraph = "\n".join(lines[start : number - 1])
        if paragraph.strip():
            first = start + 1 + next(i for i, row in enumerate(lines[start:]) if row.strip())
            label = f"line {first}" if first == number - 1 else f"lines {first}-{number - 1}"
            sections.append(Section(source, label, paragraph.strip()))
        start = number
    return sections


def _pdf_sections(source: str, data: bytes) -> list[Section]:
    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted or len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError("PDF must be unencrypted and at most 500 pages.")
    return [
        Section(source, f"page {number}", page.extract_text() or "")
        for number, page in enumerate(reader.pages, 1)
    ]


def _docx_sections(source: str, data: bytes) -> list[Section]:
    with ZipFile(BytesIO(data)) as archive:
        if sum(info.file_size for info in archive.infolist()) > MAX_DOCX_UNCOMPRESSED_BYTES:
            raise ValueError("DOCX uncompressed content exceeds 50 MB.")
    sections = []
    for number, block in enumerate(Document(BytesIO(data)).iter_inner_content(), 1):
        if isinstance(block, Table):
            text = "\n".join(" | ".join(cell.text for cell in row.cells) for row in block.rows)
        else:
            text = block.text
        sections.append(Section(source, f"block {number}", text))
    return sections


def chunk_sections(
    sections: list[Section], chunk_tokens: int = 500, overlap_tokens: int = 75
) -> list[Chunk]:
    """Token-bounded chunks: small adjacent sections are packed, large ones split with overlap."""
    if chunk_tokens < 4 or not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("Use chunk_tokens >= 4 and 0 <= overlap_tokens < chunk_tokens.")
    chunks: list[Chunk] = []
    packed: list[Section] = []
    packed_tokens = 0

    def flush() -> None:
        nonlocal packed, packed_tokens
        if packed:
            location = packed[0].location
            if len(packed) > 1:
                location = f"{location} – {packed[-1].location}"
            text = "\n\n".join(section.text.strip() for section in packed)
            chunks.append(_chunk(packed[0].source, location, 0, text))
        packed, packed_tokens = [], 0

    for section in sections:
        tokens = ENCODING.encode(section.text.strip(), disallowed_special=())
        if not tokens:
            continue
        same_source = not packed or packed[0].source == section.source
        if len(tokens) > chunk_tokens:
            flush()
            for start, text in _split(tokens, chunk_tokens, overlap_tokens):
                chunks.append(_chunk(section.source, section.location, start, text))
        elif same_source and packed_tokens + len(tokens) + 2 <= chunk_tokens:
            packed.append(section)
            packed_tokens += len(tokens) + 2
        else:
            flush()
            packed, packed_tokens = [section], len(tokens)
    flush()
    return chunks


def _chunk(source: str, location: str, start: int, text: str) -> Chunk:
    identity = f"{source}\0{location}\0{start}\0{text}"
    return Chunk(hashlib.sha256(identity.encode()).hexdigest()[:20], source, location, text)


def _split(tokens: list[int], size: int, overlap: int) -> Iterator[tuple[int, str]]:
    """Yield (start, text) windows that never cut a multi-token UTF-8 character."""
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        while True:
            try:
                text = ENCODING.decode(tokens[start:end], errors="strict")
                break
            except UnicodeDecodeError:
                end -= 1
                if end <= start:
                    raise ValueError("Chunk token budget is too small for this text.") from None
        if text.strip():
            yield start, text
        if end == len(tokens):
            return
        start = max(start + 1, end - overlap)
        while start < end:
            try:
                ENCODING.decode(tokens[start:end], errors="strict")
                break
            except UnicodeDecodeError:
                start += 1
