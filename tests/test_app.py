from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag.config import Settings
from rag.ingestion import ingest
from rag.runtime import create_agent
from tests.make_ui_fixtures import main as make_fixtures

HANDBOOK = (
    b"The reimbursement deadline is 30 days after purchase. "
    b"Submit reimbursement receipts through the expense portal."
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    for name in ("OPENAI_API_KEY", "OPENAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def run_app() -> AppTest:
    return AppTest.from_file("app.py", default_timeout=30).run()


def seed(workspace: Path, name: str = "handbook.txt", data: bytes = HANDBOOK) -> None:
    settings = Settings(data_dir=workspace)
    agent = create_agent("Offline demo", settings)
    ingest(name, data, agent.store, agent.embedder, settings)


def test_empty_demo_loads(workspace: Path) -> None:
    app = run_app()
    assert not app.exception
    assert app.selectbox(key="mode").value == "Offline demo"
    assert app.metric[1].value == "0"
    assert any("Upload a document" in info.value for info in app.info)
    assert app.button(key="index").disabled


def test_missing_key_shows_safe_error(workspace: Path) -> None:
    app = run_app()
    app.selectbox(key="mode").select("OpenAI").run()
    assert not app.exception
    assert "Set OPENAI_API_KEY" in app.error[0].value
    assert app.chat_input[0].disabled


def test_chat_citations_history_latency_and_clear(workspace: Path) -> None:
    seed(workspace)
    app = run_app()
    app.chat_input[0].set_value("What is the reimbursement deadline?").run()
    assert not app.exception
    assert any("30 days" in text.value for text in app.text)
    assert any("handbook.txt" in expander.label for expander in app.expander)
    assert app.metric[2].value.endswith("ms")
    app.chat_input[0].set_value("How do I submit it?").run()
    assert any("expense portal" in text.value for text in app.text)
    app.chat_input[0].set_value("Hello").run()
    assert any("· 0 retrievals ·" in caption.value for caption in app.caption)
    app.chat_input[0].set_value("Who is the CEO?").run()
    assert any("could not find" in text.value for text in app.text)
    assert any("Latency per question (4)" in e.label for e in app.expander)
    app.button(key="clear").click().run()
    assert not app.chat_message
    assert app.metric[1].value == "1"


def test_baseline_comparison(workspace: Path) -> None:
    seed(workspace)
    app = run_app()
    app.toggle(key="compare").set_value(True).run()
    app.chat_input[0].set_value("What is the reimbursement deadline?").run()
    assert not app.exception
    assert any(e.label.startswith("Single-shot baseline") for e in app.expander)


def test_remove_document(workspace: Path) -> None:
    seed(workspace)
    app = run_app()
    app.button(key="remove:handbook.txt").click().run()
    assert not app.exception
    assert app.metric[0].value == "0"


def test_ui_fixtures(tmp_path: Path) -> None:
    make_fixtures(["--output", str(tmp_path)])
    assert {p.name for p in tmp_path.iterdir()} == {"warranty.pdf", "support.docx", "broken.pdf"}
