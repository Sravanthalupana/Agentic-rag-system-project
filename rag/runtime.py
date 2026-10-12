from typing import Literal

from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError
from pydantic import ValidationError

from rag.agent import RAGAgent
from rag.config import Settings
from rag.demo import DemoChat, DemoEmbedder
from rag.embeddings import OpenAIEmbedder
from rag.llm import OpenAIChat
from rag.store import VectorStore

ProviderMode = Literal["OpenAI", "Offline demo"]
SAFE_MESSAGES = (
    "Supported formats:",
    "Document must",
    "PDF must",
    "DOCX uncompressed",
    "Unreadable document",
    "Could not read this document",
    "No readable text",
    "Extracted text exceeds",
    "Embedding model changed",
    "Ask a nonempty",
    "Set OPENAI_API_KEY",
)


def create_agent(mode: ProviderMode, settings: Settings) -> RAGAgent:
    """Offline demo uses its own store so lexical vectors never mix with OpenAI ones."""
    if mode == "Offline demo":
        demo = DemoEmbedder()
        store = VectorStore(settings.data_dir / "offline", demo.identity)
        return RAGAgent(store, demo, DemoChat(), settings)
    embedder = OpenAIEmbedder(settings)
    store = VectorStore(settings.data_dir, embedder.identity)
    return RAGAgent(store, embedder, OpenAIChat(settings), settings)


def safe_error(error: Exception) -> str:
    """User-facing message that never leaks provider bodies, document text, or secrets."""
    if isinstance(error, AuthenticationError):
        return "OpenAI authentication failed. Check the server's API key."
    if isinstance(error, RateLimitError):
        return "OpenAI quota or rate limit reached. Check billing or try again later."
    if isinstance(error, APIConnectionError):
        return "Could not reach OpenAI. Check connectivity and try again."
    if isinstance(error, APIStatusError):
        return "OpenAI could not complete this request. Check the model setting and retry."
    if isinstance(error, ValidationError):
        return "The model returned an invalid response. Please try again."
    if isinstance(error, ValueError) and str(error).startswith(SAFE_MESSAGES):
        return str(error)
    return "Could not complete this operation. Check the document or configuration and retry."
