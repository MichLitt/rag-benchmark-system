# AGENTS.md

When this repository is checked out inside `agent-systems-portfolio`, also read
`../AGENTS.md` and `../docs/engineering/ENGINEERING_GUIDE.md`. This file remains
the complete local entry point for a standalone clone and defines RAG-specific
rules.

## Commands

- Install: `uv sync && cp .env.example .env`
- All tests: `uv run pytest`
- Phase 2 smoke: `uv run pytest tests/test_smoke_phase2.py -q`
- API: `uv run python scripts/start_api.py --data-dir data/indexes --port 8080`
- PDF upload: `curl -X POST http://localhost:8080/v1/ingest -F "file=@docs/paper.pdf" -F "index_id=mypdf"`
- Retrieval: `curl -X POST http://localhost:8080/v1/retrieve -H 'Content-Type: application/json' -d '{"query":"...","index_id":"mypdf","top_k":5}'`
- Benchmark: `uv run python scripts/run_naive_rag_baseline.py --config config/default.yaml --dataset hotpotqa --num-queries 50`

## Modes

1. Offline benchmark: retrieval, optional expansion/reranking, generation, metrics, and failure analysis.
2. HTTP knowledge service: PDF/OCR ingestion, BM25/FAISS index discovery, and scored retrieval with provenance.
3. Citation generation/evaluation: dataset prompts, post-processing, inline citations, and optional HHEM feedback.

## Invariants

- `Document` provenance fields (`source`, `page_start`, `page_end`, `section`) must survive ingestion, storage, retrieval, and API serialization.
- Runtime ingestion and `IndexRegistry` must share the configured `--data-dir`.
- `index_id` must remain a safe portable identifier and must not allow path traversal.
- tiktoken is preferred; offline fallback must remain reversible and must not fail import or ingestion.
- API response changes require matching Agent tool changes and the root closure script.
- `rag/v1` changes require matching EvalOps schema/adapter changes.

## Environment

- `LLM_API_KEY` / `LLM_BASE_URL` — generation only
- `EVALOPS_ENDPOINT` — optional `/v1/ingest/rag/v1`
- `EVALOPS_API_KEY` — optional bearer token
- Tesseract is an optional system dependency for real OCR.

## Benchmark Integrity

- Do not cite metrics without matching run artifacts and the owning report.
- Keep synthetic/offline closure metrics separate from real dataset/model claims.
- External model downloads and paid generator calls are optional tests and must be reported separately.
- Preserve Phase 1/2 historical reports; add a new versioned report rather than rewriting evidence.

## Required Handoff

- `uv run pytest` passes; optional skips are named.
- API/ingestion changes run the Agent suite and `../scripts/run_three_project_closure.sh`.
- README commands and job statuses match the implementation (`queued → processing → completed|failed`).
- `git diff --check` passes.
