---
name: test-app
description: Install, run, and test the Agentic RAG system; mandatory before every PR.
---

# Test app

Run every command from the repository root. Read this file before **every** PR and report the
commands you ran, test counts, provider mode, and any gaps. Steps for components that do not
exist yet are marked "once X exists"; skip them until then.

## 1. Install
Python 3.11+ is required. If `python3.12` is not on PATH, look for a pyenv build
(e.g. `~/.pyenv/versions/3.12.*/bin/python3.12`).
```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```
Credentials come only from the environment: `OPENAI_API_KEY` (alias `OPENAL_API_KEY`).
Never echo, log, or commit a key. Check presence without printing it:
`.venv/bin/python -c "from rag.config import has_api_key; print(has_api_key())"`.

## 2. Quality gates (every PR)
```sh
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest -q
```
All four must pass. If formatting fails, run `.venv/bin/python -m ruff format .` and rerun.
Unit tests use fake providers; they must pass with no key and no network.

## 3. Ingestion smoke test (once `rag/ingestion.py` exists)
Use a throwaway store so real data is untouched:
```sh
RAG_DATA_DIR=/tmp/rag-smoke .venv/bin/python -m rag.ingestion --offline evals/corpus/*.txt
RAG_DATA_DIR=/tmp/rag-smoke .venv/bin/python -m rag.ingestion --offline evals/corpus/*.txt
```
Expect per-file chunk counts; the second run must not grow the total (deduplication).
With a key, drop `--offline` to use OpenAI embeddings.

## 4. Agent smoke test (once `rag/agent.py` exists)
```sh
RAG_DATA_DIR=/tmp/rag-smoke .venv/bin/python -m rag.agent --offline "What is the reimbursement deadline?"
RAG_DATA_DIR=/tmp/rag-smoke .venv/bin/python -m rag.agent --offline "Hello"
RAG_DATA_DIR=/tmp/rag-smoke .venv/bin/python -m rag.agent --offline "Who is the CEO?"
```
Expect: a cited answer containing "30 days"; a greeting with zero retrievals and no citations;
an abstention ("could not find") for the CEO. With a key, repeat without `--offline`.

## 5. Evaluation (once `evals/` exists)
```sh
.venv/bin/python -m evals.run --offline   # deterministic fakes: tests the harness only
.venv/bin/python -m evals.run             # real OpenAI comparison (paid; needs key)
```
Confirm the comparison table lists `agentic` and `baseline` with accuracy, citation validity,
source coverage, latency, and retrieval count. Never present offline scores as model quality.

## 6. Run the UI (once `app.py` exists)
```sh
RAG_DATA_DIR=/tmp/rag-ui-test .venv/bin/python -m streamlit run app.py \
  --server.address 0.0.0.0 --server.port 8501
```
Open http://localhost:8501. Generate upload fixtures with
`.venv/bin/python -m tests.make_ui_fixtures` (writes to `artifacts/ui-fixtures/`).

## 7. Recorded browser checks (mandatory for every PR touching the UI)
Record the screen while performing these in the browser, then attach the video to the PR
(never commit media):
1. App loads; provider mode is shown explicitly; empty corpus shows getting-started help.
2. Upload `evals/corpus/handbook.txt`, click **Index documents**; success + chunk count shown.
3. Index the same file again; the chunk count does not grow.
4. Ask "What is the reimbursement deadline?"; answer mentions 30 days, has an expandable
   citation (file, location, chunk text), a retrieval trace, and latency. Ask the follow-up
   "How do I submit it?" and expect the expense portal.
5. Ask "Hello"; zero retrievals, no citations.
6. Ask "Who is the CEO?"; the agent says the documents do not contain it.
7. Upload the generated `warranty.pdf` and `support.docx`; ask about their facts.
8. Upload `broken.pdf`; a safe error appears and the app keeps working.
9. Latency panel shows per-question timings; reload keeps indexed documents;
   **Clear conversation** empties chat but keeps documents.

Without a key, use the sidebar's **Offline demo** provider: this validates wiring only, so
report the real-OpenAI check as a gap. With a key, repeat steps 2 and 4 in OpenAI mode.
