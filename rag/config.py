import os
from dataclasses import dataclass
from pathlib import Path

KEY_NAMES = ("OPENAI_API_KEY", "OPENAL_API_KEY")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(".data")
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_tokens: int = 500
    overlap_tokens: int = 75
    top_k: int = 4
    max_retrievals: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            data_dir=Path(os.environ.get("RAG_DATA_DIR", str(defaults.data_dir))),
            chat_model=os.environ.get("RAG_CHAT_MODEL", defaults.chat_model),
            embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", defaults.embedding_model),
            chunk_tokens=_int_env("RAG_CHUNK_TOKENS", defaults.chunk_tokens),
            overlap_tokens=_int_env("RAG_OVERLAP_TOKENS", defaults.overlap_tokens),
            top_k=_int_env("RAG_TOP_K", defaults.top_k),
            max_retrievals=_int_env("RAG_MAX_RETRIEVALS", defaults.max_retrievals),
        )


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer.") from None


def has_api_key() -> bool:
    return any(os.environ.get(name) for name in KEY_NAMES)


def api_key() -> str:
    for name in KEY_NAMES:
        if key := os.environ.get(name):
            return key
    raise ValueError("Set OPENAI_API_KEY (or OPENAL_API_KEY), or select Offline demo.")
