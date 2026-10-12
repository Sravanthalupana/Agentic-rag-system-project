from pathlib import Path

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError, RateLimitError

from rag.config import Settings
from rag.runtime import create_agent, safe_error


def test_offline_mode_uses_separate_store(tmp_path: Path) -> None:
    agent = create_agent("Offline demo", Settings(data_dir=tmp_path))
    assert agent.store.path.parent == tmp_path / "offline"
    assert agent.ask("Hello").retrieval_count == 0


def test_safe_error_hides_provider_details() -> None:
    response = httpx.Response(401, request=httpx.Request("POST", "https://example.com"))
    auth = AuthenticationError("sensitive provider body", response=response, body=None)
    assert "authentication" in safe_error(auth) and "sensitive" not in safe_error(auth)
    assert "quota" in safe_error(RateLimitError("sensitive", response=response, body=None))
    assert "reach OpenAI" in safe_error(APIConnectionError(request=response.request))
    assert "sensitive" not in safe_error(ValueError("sensitive generated content"))
    assert "retry" in safe_error(RuntimeError("secret value"))
    assert safe_error(ValueError("Supported formats: PDF")) == "Supported formats: PDF"


def test_openai_mode_requires_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "OPENAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="Set OPENAI_API_KEY"):
        create_agent("OpenAI", Settings(data_dir=tmp_path))
