from pathlib import Path

import pytest

from rag.config import Settings, api_key, has_api_key


def test_defaults() -> None:
    settings = Settings()
    assert settings.chat_model == "gpt-4o-mini"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.overlap_tokens < settings.chunk_tokens


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_DATA_DIR", "/tmp/custom-rag")
    monkeypatch.setenv("RAG_CHAT_MODEL", "test-model")
    monkeypatch.setenv("RAG_TOP_K", "7")
    settings = Settings.from_env()
    assert settings.data_dir == Path("/tmp/custom-rag")
    assert settings.chat_model == "test-model"
    assert settings.top_k == 7


def test_invalid_integer_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "many")
    with pytest.raises(ValueError, match="RAG_TOP_K"):
        Settings.from_env()


def test_key_alias_and_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAL_API_KEY", "fake-alias")
    assert has_api_key() and api_key() == "fake-alias"
    monkeypatch.setenv("OPENAI_API_KEY", "fake-standard")
    assert api_key() == "fake-standard"


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "OPENAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert not has_api_key()
    with pytest.raises(ValueError, match="Set OPENAI_API_KEY"):
        api_key()
