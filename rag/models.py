from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """A located span of extracted text, e.g. one PDF page or DOCX block."""

    source: str
    location: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    location: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
