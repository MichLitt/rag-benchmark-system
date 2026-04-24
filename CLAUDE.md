# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # install dependencies
uv run pytest tests/ -q          # run all tests (316 pass; test_query_factory.py has 2 pre-existing failures needing LLM_BASE_URL mock — safe to ignore)
uv run pytest tests/test_smoke_phase2.py -q   # run Phase 2 integration smoke tests
uv run pytest tests/<file> -k "<name>" -q     # run a single test

# Retrieval API server (indexes registered via INDEX_CONFIG_<ID> env vars)
INDEX_CONFIG_DEFAULT=config/wiki18_21m_sharded.yaml uv run python scripts/start_api.py --port 8080

# Async ingest endpoint (upload PDF, poll for completion)
curl -X POST http://localhost:8080/v1/ingest -F "file=@docs/paper.pdf" -F "index_id=mypdf"
curl http://localhost:8080/v1/ingest/<job_id>   # poll: pending → running → complete

# PDF ingestion (offline, before building index)
uv run python scripts/ingest_documents.py --input docs/paper.pdf --output data/indexes/mypdf/docstore.jsonl

# Build retrieval index from docstore
uv run python scripts/build_retrieval_indexes.py --index-dir data/indexes/mypdf/

# Run a benchmark eval
uv run python scripts/run_naive_rag_baseline.py --config config/default.yaml --dataset hotpotqa --num-queries 50

# Score citation / faithfulness post-hoc
uv run python scripts/score_citation.py --input experiments/run_results.json --output experiments/run_results_nli.json
uv run python scripts/score_faithfulness.py --matrix-dir experiments/runs/phase4_matrix/
```

## Environment Variables

```
LLM_API_KEY        — required for generation (OpenAI-compatible endpoint)
LLM_BASE_URL       — required for generation (e.g. https://api.minimax.io/v1)
EVALOPS_ENDPOINT   — optional; set to http://<host>/v1/ingest/rag/v1 to report eval runs to llm-evalops-platform
EVALOPS_API_KEY    — optional bearer token for EvalOps platform
```

## Architecture

This project has two distinct modes that share the same retrieval and evaluation infrastructure:

**Mode 1 — Offline Benchmark** (`scripts/run_naive_rag_baseline.py`, `main.py`)
The pipeline is Query → Retriever → (optional QueryExpander → re-Retriever) → (optional Reranker) → Generator → EM/F1/Recall metrics. Configured entirely via YAML (`config/`). Results go to `experiments/` as JSON + JSONL trajectories.

**Mode 2 — HTTP Retrieval Service** (`src/api/`)
A FastAPI server exposing `POST /v1/retrieve` and `GET /v1/health`. Indexes are discovered automatically from `data/indexes/` sub-directories (look for `docstore.jsonl`). Dense retrieval needs `index.faiss` + `dense_config.json`; BM25 needs `bm25.pkl`. Dense takes priority when both exist. The `IndexRegistry` lazily loads retrievers on first access under per-index locks (thread-safe).

### Key Abstractions

- **`Document`** (`src/types.py`) — core unit. Carries `page_start/page_end/source/section` for PDF provenance; `extra_metadata` dict for arbitrary extension.
- **`RunExampleResult`** (`src/types.py`) — per-query result record with full retrieval, generation, and citation metrics. The EvalOps adapter reads from this.
- **`IndexRegistry`** (`src/api/index_registry.py`) — thread-safe lazy-loader. Auto-detects retriever type from files on disk. The global `_registry` in `handlers.py` is the shared instance; `set_registry()` replaces it in tests.
- **Pipeline** (`src/pipeline.py`) — orchestrates a single query: retrieval → optional expansion → optional reranking → generation → metrics. Called by both the CLI runner and eval scripts.

### Ingestion Pipeline (PDF → retrievable index)

Two-stage: `Parser` → `TokenAwareChunker` (tiktoken sliding window, 256 tokens / 32 overlap). Chunker preserves `page_start/page_end` by maintaining a per-token page-number map. Produces `docstore.jsonl`.

The factory (`src/ingestion/factory.py`) selects the parser automatically: if pdfplumber extracts fewer than 10 characters from the first page, it falls back to `OcrParser` (PyMuPDF render → Tesseract OCR). Tesseract must be installed on the system (`brew install tesseract` / `apt-get install tesseract-ocr`).

### EvalOps Integration

`src/evalops/` contains a three-layer design: `schema.py` (EvalRunReport dataclass) → `adapter.py` (builds report from RunExampleResult list + metrics dict) → `client.py` (fire-and-forget POST to `EVALOPS_ENDPOINT`). The client is already wired into `scripts/run_naive_rag_baseline.py`. When `EVALOPS_ENDPOINT` is unset, the client is a no-op.

### Retrieval Options

| Mode | Files needed | Notes |
|------|-------------|-------|
| FAISS dense | `index.faiss`, `dense_config.json`, `docstore.jsonl` | all-MiniLM-L6-v2 embeddings |
| BM25 sparse | `bm25.pkl`, `docstore.jsonl` | rank-bm25 |
| Hybrid | both | weighted RRF fusion in `src/retrieval/hybrid.py` |
| Reranker | either + model download | cross-encoder/ms-marco-MiniLM-L-6-v2; requested via `use_reranker=true` in API |

### Test Layout

Phase 1 tests cover retrieval components, pipeline, metrics, and query expansion. Phase 2 smoke tests (`test_smoke_phase2.py`, 26 cases) cover the full A0/A1/A2/A3 stack using FastAPI `TestClient` and in-memory PDF generation — no real index or model download needed. `test_docstore_migration.py` guards schema backward compatibility.

## Phase Status

- **Phase 1** (closed 2026-03-10): Offline benchmark on HotpotQA/NQ/TriviaQA. Key finding: Recall@k is 0.63–0.81; generation is the bottleneck (F1 0.10–0.21).
- **Phase 2** (closed 2026-04-08): PDF ingestion, FastAPI retrieval API, NLI citation evaluation. See `report/phase2_closure_20260408.md` for full delivery list and known limitations.
- **Phase 3** (closed 2026-04-24): OCR/scanned-PDF pipeline (`src/ingestion/ocr_parser.py`), async `/v1/ingest` upload endpoint (`src/api/ingest.py`), CLAUDE.md added. 339 tests passing.
