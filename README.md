# RAG Benchmark System → Agent Knowledge Subsystem

Modular RAG benchmark system that evaluates retrieval strategies on standard QA benchmarks,
now extended into a **real-document knowledge service** with PDF ingestion, a FastAPI retrieval
API, and NLI-based citation evaluation.

## Architecture

```
Phase 1 (benchmark)                 Phase 2 (agent subsystem)
──────────────────                  ─────────────────────────
Query → Retriever → Reranker   →   FastAPI /v1/retrieve
         ↓                                  ↓
      Generator                     PDF Ingestion Pipeline
         ↓                                  ↓
   EM / F1 / Recall             NLI Citation Evaluation (HHEM)
```

### Key Components

| Component | Implementation | Details |
|-----------|---------------|---------|
| Dense Retrieval | FAISS (flat / sharded) | 21M Wikipedia passages, all-MiniLM-L6-v2 |
| Sparse Retrieval | rank-bm25 | BM25 index for hybrid search |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder with rank fusion blending |
| Query Expansion | HyDE, HotpotQA Decompose | LLM-based query rewriting |
| Generation | OpenAI-compatible API | MiniMax-M2.5 with reasoning split |
| **PDF Ingestion** | pdfplumber + tiktoken | Native-text PDF → token-aware chunks with page metadata |
| **Retrieval API** | FastAPI + uvicorn | `/v1/retrieve` and `/v1/health` HTTP endpoints |
| **NLI Evaluation** | Vectara HHEM | Post-hoc citation attribution (answer_attribution_rate, page_grounding_accuracy) |
| Evaluation | EM, F1, Recall@k, Faithfulness | LLM-as-judge + NLI-as-judge |
| Analysis | Failure mode classification | HotpotQA taxonomy + generic failure modes |

---

## Phase 1: Benchmark Results

5 configurations evaluated across 3 datasets with 200 queries each:

| Config | Retriever | Reranker | Query Expansion |
|--------|-----------|----------|-----------------|
| C1 Dense Only | dense_sharded | off | off |
| C2 Dense+Rerank | dense_sharded | on | off |
| C3 Dense+Rerank (top-5) | dense_sharded | on | off (narrow context) |
| C4 Rerank+HyDE | dense_sharded | on | HyDE |
| C5 Rerank+Decompose | dense_sharded | on | auto |

| Config | HotpotQA F1 | NQ F1 | TriviaQA F1 | Avg F1 | Cost/200q |
|--------|------------|-------|------------|--------|-----------|
| C1 Dense Only | 0.101 | 0.116 | 0.176 | 0.131 | $0.35 |
| C2 Dense+Rerank | 0.131 | 0.112 | **0.212** | 0.152 | $0.34 |
| C3 Dense+Rerank (top-5) | 0.138 | 0.113 | 0.187 | 0.146 | **$0.16** |
| C4 Rerank+HyDE | **0.145** | 0.108 | 0.194 | 0.149 | $0.34 |
| C5 Rerank+Decompose | 0.123 | **0.126** | 0.190 | 0.146 | $0.35 |

**Key Finding**: Generation is the bottleneck — Recall@k reaches 0.63–0.81 but F1 stays 0.10–0.21.

---

## Phase 2: Agent Knowledge Subsystem

### PDF Ingestion

```bash
# Ingest PDF documents into a JSONL docstore
uv run python scripts/ingest_documents.py \
    --input docs/manual.pdf docs/spec.pdf \
    --output data/indexes/my_index/docstore.jsonl \
    --chunk-size 256 \
    --overlap 32
```

Each chunk carries `page_start`, `page_end`, `source`, and `section` metadata for downstream citation grounding.

### Retrieval API

```bash
# Start the API server (register indexes via env vars)
INDEX_CONFIG_DEFAULT=config/wiki18_21m_sharded.yaml \
    uv run python scripts/start_api.py --port 8080

# Health check
curl http://localhost:8080/v1/health

# Retrieve
curl -X POST http://localhost:8080/v1/retrieve \
    -H "Content-Type: application/json" \
    -d '{"query": "who founded Apple", "top_k": 5, "index_id": "default"}'
```

Response includes `page_start`, `page_end`, `source`, and `section` fields when available.

Multiple indexes can be loaded simultaneously:
```bash
INDEX_CONFIG_WIKI=config/wiki18_21m_sharded.yaml \
INDEX_CONFIG_MYPDF=config/my_pdf_index.yaml \
    uv run python scripts/start_api.py
```

### NLI Citation Evaluation

```bash
# Score existing run results with NLI attribution
uv run python scripts/score_citation.py \
    --input experiments/phase4_results.json \
    --output experiments/phase4_results_nli.json

# Run full Phase 2 evaluation
uv run python scripts/eval_phase2_full.py \
    --output report/phase2_eval.json

# With real Vectara HHEM model (~500MB download on first run)
uv run python scripts/eval_phase2_full.py --use-real-hhem
```

**NLI Metrics**:
- `answer_attribution_rate` — fraction of retrieved passages consistent with the answer
- `supporting_passage_hit` — any passage above the NLI consistency threshold
- `page_grounding_accuracy` — of consistent passages, fraction with page metadata (PDF only)

---

## Phase 1 Quick Start

```bash
# Install dependencies
uv sync

# Set API keys
export LLM_API_KEY="your_api_key"
export LLM_BASE_URL="https://api.minimax.io/v1"

# Run the full experiment matrix (dry run first)
uv run python scripts/run_phase4_matrix.py --dry-run
uv run python scripts/run_phase4_matrix.py

# Score faithfulness on results
uv run python scripts/score_faithfulness.py --matrix-dir experiments/runs/phase4_matrix/

# Aggregate results
uv run python scripts/aggregate_experiment_results.py --matrix-dir experiments/runs/phase4_matrix/

# Analyze failure modes
uv run python scripts/analyze_cross_config_failures.py --matrix-dir experiments/runs/phase4_matrix/

# Export dashboard data and launch
uv run python scripts/export_dashboard_data.py --matrix-dir experiments/runs/phase4_matrix/
uv run streamlit run app/dashboard.py
```

---

## Evaluation Metrics

### Phase 1 (QA Benchmark)
- **EM (Exact Match)**: Normalized string equality
- **F1**: Token-level overlap score
- **Recall@k**: Gold document presence in top-k results
- **Faithfulness (LLM-as-judge)**: LLM scores answer support by context (0–1)
- **Hallucination Rate**: Fraction of answers with faithfulness < 0.5

### Phase 2 (Citation / Page Grounding)
- **answer_attribution_rate**: Post-hoc NLI — fraction of retrieved passages that entail the answer
- **supporting_passage_hit**: Any retrieved passage above the NLI consistency threshold
- **page_grounding_accuracy**: Fraction of consistent passages with `page_start`/`page_end` metadata

---

## Project Structure

```
rag-benchmark-study/
├── app/
│   └── dashboard.py            # Streamlit results dashboard
├── config/
│   ├── phase4/                 # Phase 4 experiment configs
│   └── *.yaml                  # Other experiment configs
├── data/
│   ├── raw/                    # Downloaded QA sets and corpus
│   ├── indexes/                # Built retrieval indexes
│   └── filtered/               # Filtered subsets
├── experiments/
│   ├── runs/                   # All experiment outputs
│   ├── cache/                  # Query expansion cache
│   └── phase4_results.*        # Aggregated results
├── report/
│   ├── charts/                 # Static chart images
│   ├── phase2_eval.json        # Phase 2 evaluation metrics
│   └── *.md                    # Analysis and closure reports
├── scripts/
│   ├── run_phase4_matrix.py    # Batch experiment runner
│   ├── score_faithfulness.py   # Post-hoc LLM faithfulness scoring
│   ├── score_citation.py       # Post-hoc NLI citation scoring (Phase 2)
│   ├── ingest_documents.py     # PDF ingestion CLI (Phase 2)
│   ├── start_api.py            # FastAPI server launcher (Phase 2)
│   ├── eval_phase2_full.py     # Phase 2 full evaluation (Phase 2)
│   └── ...
├── src/
│   ├── retrieval/              # Retriever implementations
│   ├── reranking/              # Cross-encoder reranker
│   ├── generation/             # LLM generation backends
│   ├── query/                  # Query expansion modules
│   ├── evaluation/             # Metrics + faithfulness + NLI citation
│   ├── analysis/               # Failure mode classifiers
│   ├── ingestion/              # PDF parsing + chunking (Phase 2)
│   ├── api/                    # FastAPI service (Phase 2)
│   └── pipeline.py             # End-to-end RAG pipeline
├── tests/                      # Unit + smoke + migration tests
├── Dockerfile                  # Dashboard container
└── pyproject.toml
```

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Embedding | all-MiniLM-L6-v2 (SentenceTransformers) |
| Dense Index | FAISS (flat, sharded) |
| Sparse Index | rank-bm25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | MiniMax-M2.5 (OpenAI-compatible) |
| Datasets | HotpotQA, NQ, TriviaQA (FlashRAG) |
| Dashboard | Streamlit + Plotly |
| **PDF Parsing** | pdfplumber |
| **API** | FastAPI + uvicorn |
| **NLI Scorer** | Vectara HHEM (transformers) |
| Package Manager | uv |
