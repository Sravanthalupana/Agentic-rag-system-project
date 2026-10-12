from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rag.config import Settings
from rag.demo import DemoEmbedder
from rag.embeddings import OpenAIEmbedder


def test_openai_embedder_orders_and_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-not-real")
    with patch("rag.embeddings.OpenAI") as client:
        create = client.return_value.embeddings.create
        create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ]
        )
        embedder = OpenAIEmbedder(Settings())
        assert embedder.identity == "openai:text-embedding-3-small"
        assert embedder.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]
        create.reset_mock()
        embedder.embed(["text"] * 129)
        assert [len(call.kwargs["input"]) for call in create.call_args_list] == [128, 1]
        create.reset_mock()
        assert embedder.embed([]) == []
        create.assert_not_called()


def test_openai_embedder_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "OPENAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="Set OPENAI_API_KEY"):
        OpenAIEmbedder(Settings())


def test_demo_embedder_is_deterministic_and_lexical() -> None:
    embedder = DemoEmbedder()
    first, second, other = embedder.embed(["refund window", "refund window", "backup region"])
    assert first == second and first != other
    assert any(embedder.embed(["the a of"])[0])
