from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from rag.config import Settings
from rag.llm import OpenAIChat


def reply(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def create(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-not-real")
    with patch("rag.llm.OpenAI") as client:
        yield client.return_value.chat.completions.create


def test_json_contract(create: MagicMock) -> None:
    create.return_value = reply('{"action":"retrieve","queries":["expense policy"]}')
    decision = OpenAIChat(Settings()).complete("Return JSON", {"question": "test"})
    assert decision.queries == ["expense policy"]
    kwargs = create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"} and kwargs["temperature"] == 0
    assert kwargs["messages"][1] == {"role": "user", "content": '{"question": "test"}'}


def test_repairs_one_invalid_reply(create: MagicMock) -> None:
    create.side_effect = [reply("not json"), reply('{"action":"answer","answer":"ok"}')]
    assert OpenAIChat(Settings()).complete("JSON", {}).answer == "ok"
    assert create.call_count == 2
    assert create.call_args.kwargs["messages"][-1]["role"] == "user"


@pytest.mark.parametrize("content", [None, "not json", '{"action":"execute_shell"}'])
def test_rejects_repeated_invalid_replies(create: MagicMock, content: str | None) -> None:
    create.return_value = reply(content)
    with pytest.raises(ValidationError):
        OpenAIChat(Settings()).complete("JSON", {})
    assert create.call_count == 2
