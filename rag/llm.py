import json
from typing import Literal, Protocol

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag.config import Settings, api_key


class Decision(BaseModel):
    """One agent step: either search (with 1-3 queries) or answer with cited chunk IDs."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["retrieve", "answer"]
    queries: list[str] = Field(default_factory=list, max_length=3)
    answer: str = Field(default="", max_length=8000)
    citations: list[str] = Field(default_factory=list, max_length=12)
    abstain: bool = False
    document_question: bool = True
    reason: str = Field(default="", max_length=500)


class ChatModel(Protocol):
    def complete(self, system: str, payload: dict[str, object]) -> Decision: ...


class OpenAIChat:
    """JSON-mode chat completion validated against `Decision`, with one repair retry."""

    def __init__(self, settings: Settings) -> None:
        self.model = settings.chat_model
        self.client = OpenAI(api_key=api_key(), timeout=45.0, max_retries=2)

    def complete(self, system: str, payload: dict[str, object]) -> Decision:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        for attempt in range(2):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=1800,
            )
            content = response.choices[0].message.content or ""
            try:
                return Decision.model_validate_json(content)
            except ValidationError:
                if attempt:
                    raise
                messages += [
                    {"role": "assistant", "content": content[:4000]},
                    {
                        "role": "user",
                        "content": "That was not a valid JSON object with the required fields. "
                        "Reply again with only the JSON object.",
                    },
                ]
        raise AssertionError("unreachable")
