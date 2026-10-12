from io import BytesIO

import pytest
from pypdf import PdfWriter

from rag.documents import ENCODING, MAX_FILE_BYTES, chunk_sections, load_document
from rag.models import Section
from tests.fixtures import docx_bytes, pdf_bytes


def test_text_strips_bom_and_path() -> None:
    assert load_document("../dir\\notes.TXT", b"\xef\xbb\xbfHello") == [
        Section("notes.TXT", "line 1", "Hello")
    ]
    assert load_document("readme.md", b"# Title")[0].source == "readme.md"


def test_text_paragraphs_have_line_ranges() -> None:
    text = b"Title\n\n\nFirst line\nsecond line\n  \nLast\n"
    assert load_document("a.txt", text) == [
        Section("a.txt", "line 1", "Title"),
        Section("a.txt", "lines 4-5", "First line\nsecond line"),
        Section("a.txt", "line 7", "Last"),
    ]


def test_pdf_pages_have_locations() -> None:
    sections = load_document("guide.pdf", pdf_bytes("Warranty lasts 2 years.", "Page two."))
    assert [section.location for section in sections] == ["page 1", "page 2"]
    assert "2 years" in sections[0].text


def test_docx_paragraphs_and_tables() -> None:
    sections = load_document("guide.docx", docx_bytes())
    assert sections[0].location == "block 1"
    assert "9 AM" in sections[0].text
    assert sections[1].text == "Escalation | Contact Mira"


@pytest.mark.parametrize(
    "filename,data,message",
    [
        ("x.exe", b"x", "Supported formats"),
        ("x.txt", b"", "nonempty"),
        ("x.txt", b"\xff\xfe\xfa", "UTF-8"),
        ("x.pdf", b"not a pdf", "Could not read"),
        ("x.docx", b"not a zip", "Unreadable"),
        ("x.txt", b"   \n", "No readable text"),
    ],
)
def test_rejects_invalid_documents(filename: str, data: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_document(filename, data)


def test_size_limit_and_scanned_pdf() -> None:
    with pytest.raises(ValueError, match="20 MB"):
        load_document("large.txt", b"x" * (MAX_FILE_BYTES + 1))
    writer = PdfWriter()
    writer.add_blank_page(width=10, height=10)
    output = BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="OCR"):
        load_document("scan.pdf", output.getvalue())


def test_long_section_split_with_overlap_and_stable_ids() -> None:
    section = Section("a.txt", "text", " ".join(f"w{i}" for i in range(80)))
    chunks = chunk_sections([section], 12, 3)
    assert len(chunks) > 1
    assert all(len(ENCODING.encode(chunk.text)) <= 12 for chunk in chunks)
    assert ENCODING.encode(chunks[0].text)[-3:] == ENCODING.encode(chunks[1].text)[:3]
    assert chunks == chunk_sections([section], 12, 3)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_small_sections_are_packed_per_source() -> None:
    sections = [
        Section("a.docx", "block 1", "Alpha."),
        Section("a.docx", "block 2", "Beta."),
        Section("b.docx", "block 1", "Gamma."),
    ]
    chunks = chunk_sections(sections, 50, 5)
    assert [(c.source, c.location) for c in chunks] == [
        ("a.docx", "block 1 – block 2"),
        ("b.docx", "block 1"),
    ]
    assert chunks[0].text == "Alpha.\n\nBeta."


def test_packing_respects_budget() -> None:
    sections = [Section("a.txt", f"block {i}", "one two three four five") for i in range(6)]
    chunks = chunk_sections(sections, 16, 2)
    assert len(chunks) > 1
    assert all(len(ENCODING.encode(chunk.text)) <= 16 for chunk in chunks)


def test_unicode_and_special_tokens_survive() -> None:
    text = "世界🌍 " * 20 + "<|endoftext|>"
    chunks = chunk_sections([Section("a.txt", "text", text)], 12, 0)
    assert "".join(chunk.text for chunk in chunks) == text.strip()
    assert all("\ufffd" not in chunk.text for chunk in chunks)


@pytest.mark.parametrize("size,overlap", [(10, 10), (3, 0), (10, -1)])
def test_invalid_chunk_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_sections([], size, overlap)
