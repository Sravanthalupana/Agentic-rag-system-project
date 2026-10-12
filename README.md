# Agentic RAG System

Local-first agentic retrieval-augmented generation over your own PDF, DOCX, and text documents.
Vectors are stored locally with FAISS; OpenAI provides embeddings and the chat model.

## Quick start
```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OPENAI_API_KEY=...   # never commit this
```

## Use it
```sh
.venv/bin/python -m streamlit run app.py          # chat UI: upload, ask, citations, latency
.venv/bin/python -m rag.ingestion docs/*.pdf      # index from the CLI
.venv/bin/python -m rag.agent "What is the reimbursement deadline?"
.venv/bin/python -m evals.run                     # agentic vs single-shot baseline table
```
Add `--offline` to the CLIs (or pick **Offline demo** in the UI) to run without a key using
deterministic lexical fakes; that checks the wiring, not answer quality.

See [AGENTS.md](AGENTS.md) for structure and conventions and
[.agents/skills/test-app/SKILL.md](.agents/skills/test-app/SKILL.md) for install, run, and
test instructions.

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` (alias `OPENAL_API_KEY`) | — | OpenAI credentials |
| `RAG_DATA_DIR` | `.data` | Local vector store directory |
| `RAG_CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `RAG_CHUNK_TOKENS` / `RAG_OVERLAP_TOKENS` | `500` / `75` | Chunking |
| `RAG_TOP_K` / `RAG_MAX_RETRIEVALS` | `4` / `3` | Retrieval breadth / agent budget |
