from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import streamlit as st

from rag.agent import AgentResult, RAGAgent
from rag.config import Settings, has_api_key
from rag.documents import SUPPORTED_SUFFIXES
from rag.ingestion import ingest
from rag.runtime import ProviderMode, create_agent, safe_error

MODES: list[ProviderMode] = ["OpenAI", "Offline demo"]


@st.cache_resource(show_spinner=False)
def cached_agent(mode: ProviderMode, data_dir: Path) -> RAGAgent:
    """One agent per provider and store; the store refreshes itself when its revision changes."""
    return create_agent(mode, Settings.from_env())


def render_result(result: AgentResult, baseline: AgentResult | None = None) -> None:
    st.text(result.answer)
    st.caption(
        f"{result.latency_ms:,.0f} ms total · LLM {result.llm_ms:,.0f} ms · "
        f"retrieval {result.retrieval_ms:,.0f} ms · {result.retrieval_count} retrievals · "
        f"{'Insufficient evidence' if result.abstained else 'Answered'}"
    )
    for i, hit in enumerate(result.citations, 1):
        with st.expander(f"[{i}] {hit.chunk.source} · {hit.chunk.location}"):
            st.caption(f"Chunk {hit.chunk.id} · cosine score {hit.score:.3f}")
            st.code(hit.chunk.text, language=None, wrap_lines=True)
    with st.expander("Retrieval trace"):
        for step in result.trace:
            st.text(f"{step.action.upper()} · {step.detail} · {step.ms:,.0f} ms")
            if step.chunk_ids:
                st.caption("Chunks: " + ", ".join(step.chunk_ids))
    if baseline is not None:
        with st.expander(f"Single-shot baseline · {baseline.latency_ms:,.0f} ms"):
            st.text(baseline.answer)
            sources = ", ".join(
                f"{h.chunk.source} ({h.chunk.location})" for h in baseline.citations
            )
            st.caption(f"1 retrieval on the raw question · sources: {sources or 'none'}")


def latency_rows(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for message in messages:
        result: AgentResult | None = message.get("result")
        if result is None:
            continue
        baseline: AgentResult | None = message.get("baseline")
        rows.append(
            {
                "Question": message["question"],
                "Total ms": round(result.latency_ms),
                "LLM ms": round(result.llm_ms),
                "Retrieval ms": round(result.retrieval_ms),
                "Retrievals": result.retrieval_count,
                "Baseline ms": round(baseline.latency_ms) if baseline else None,
            }
        )
    return rows


def sidebar(settings: Settings) -> tuple[ProviderMode, RAGAgent | None, bool]:
    with st.sidebar:
        st.title("Document workspace")
        mode = st.selectbox("Provider", MODES, index=0 if has_api_key() else 1, key="mode")
        if mode == "Offline demo":
            st.warning("OFFLINE DEMO: lexical search and extractive answers. Not OpenAI quality.")
        else:
            st.caption(f"{settings.chat_model} · {settings.embedding_model}")
        agent = None
        try:
            agent = cached_agent(mode, settings.data_dir)
        except Exception as error:
            st.error(safe_error(error))
        uploads = st.file_uploader(
            "Upload documents",
            type=[suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES],
            accept_multiple_files=True,
            help="PDF, DOCX, TXT, MD · 20 MB per file",
        )
        if st.button(
            "Index documents", type="primary", disabled=not uploads or agent is None, key="index"
        ):
            assert agent is not None
            for upload in uploads or []:
                started = perf_counter()
                try:
                    with st.spinner(f"Indexing {upload.name}…"):
                        result = ingest(
                            upload.name, upload.getvalue(), agent.store, agent.embedder, settings
                        )
                    ms = (perf_counter() - started) * 1000
                    if result.skipped:
                        st.info(f"{result.source}: unchanged, already indexed ({ms:,.0f} ms)")
                    else:
                        st.success(
                            f"{result.source}: {result.chunks} chunks indexed ({ms:,.0f} ms)"
                        )
                except Exception as error:
                    st.error(f"{upload.name}: {safe_error(error)}")
        if agent is not None and agent.store.count:
            st.subheader("Indexed documents")
            for source, chunks in agent.store.sources().items():
                name, remove = st.columns([4, 1])
                name.text(f"{source} · {chunks} chunks")
                if remove.button("✕", key=f"remove:{source}", help=f"Remove {source}"):
                    agent.store.delete(source)
                    st.rerun()
            st.caption(
                "Identical files are skipped; a changed file with the same name replaces it."
            )
        compare = st.toggle(
            "Also run single-shot baseline", key="compare", help="Shows naive RAG for comparison"
        )
        if st.button("Clear conversation", key="clear"):
            st.session_state.messages = []
        st.caption(
            "Vectors and text stay on this server; OpenAI mode sends text to OpenAI. "
            "Local trusted use only: no authentication or per-user isolation."
        )
    return mode, agent, compare


def main() -> None:
    st.set_page_config(page_title="Agentic RAG · Document chat", page_icon="📚", layout="wide")
    settings = Settings.from_env()
    st.session_state.setdefault("messages", [])
    mode, agent, compare = sidebar(settings)
    if st.session_state.get("provider_mode") != mode:
        st.session_state.messages = []
        st.session_state.provider_mode = mode
    messages: list[dict[str, Any]] = st.session_state.messages

    st.title("Agentic RAG")
    st.caption("The agent decides when and what to retrieve, and cites the chunks it used.")
    if mode == "Offline demo":
        st.info(
            "Offline demo is active. For real embeddings and an LLM, set OPENAI_API_KEY "
            "on the server and select OpenAI."
        )
    rows = latency_rows(messages)
    columns = st.columns(4)
    columns[0].metric("Documents", len(agent.store.sources()) if agent else 0)
    columns[1].metric("Source chunks", agent.store.count if agent else 0)
    columns[2].metric("Last answer", f"{rows[-1]['Total ms']:,} ms" if rows else "—")
    columns[3].metric(
        "Mean latency", f"{mean(r['Total ms'] for r in rows):,.0f} ms" if rows else "—"
    )
    if rows:
        with st.expander(f"Latency per question ({len(rows)})"):
            st.dataframe(rows, hide_index=True, use_container_width=True)
    if agent and not agent.store.count:
        st.info("Upload a document in the sidebar, then click Index documents to get started.")

    for message in messages:
        with st.chat_message(message["role"]):
            if "result" in message:
                render_result(message["result"], message.get("baseline"))
            else:
                st.text(message["content"])

    question = st.chat_input(
        "Ask a question about your documents…", max_chars=2000, disabled=agent is None
    )
    if question and agent is not None:
        history = [{"role": m["role"], "content": m["content"]} for m in messages]
        with st.chat_message("user"):
            st.text(question)
        try:
            with st.spinner("Deciding what to retrieve…"):
                result = agent.ask(question, history)
                baseline = agent.baseline(question, history) if compare else None
        except Exception as error:
            st.error(safe_error(error))
            return
        messages.append({"role": "user", "content": question})
        messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "question": question,
                "result": result,
                "baseline": baseline,
            }
        )
        st.rerun()


main()
