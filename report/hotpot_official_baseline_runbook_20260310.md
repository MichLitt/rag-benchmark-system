# Hotpot Official Baseline Runbook

## 1. Official Retrieval Baseline

This document freezes the official Hotpot baseline for the retrieval-first phase and the closure-safe end-to-end sanity path.

Official retrieval config:

- [wiki18_21m_sharded_official_baseline.yaml](D:/Project/Toy/RAG-Benchmark-Study/config/wiki18_21m_sharded_official_baseline.yaml)

Frozen retrieval settings:

- `retriever = dense_sharded`
- `retrieve_top_k = 100`
- `top_k = 20`
- `dedup_mode = title`
- `dedup_before_rerank = true`
- `use_reranker = true`
- `title_first_rerank = false`
- `reranker_retriever_rank_weight = 0.4`
- `query_expansion = off`

Backbone artifact:

- `E:/rag-benchmark-indexes/wiki18_21m_dense_sharded/manifest.json`

This remains the official retrieval baseline because it is the most stable route with full-stage evidence and the best confirmed `coverage-filtered` retrieval result.

---

## 2. Standard Commands

### 2.1 200-query official retrieval control

```powershell
python scripts/run_naive_rag_baseline.py `
  --config config/wiki18_21m_sharded_official_baseline.yaml `
  --datasets hotpotqa `
  --max-queries 200 `
  --use-reranker `
  --reranker-retriever-rank-weight 0.4 `
  --output-root experiments/runs/hotpot_closure_retrieval_recheck `
  --progress-every 10 `
  --progress-min-seconds 30
```

### 2.2 Coverage-filtered full retrieval-only validation

```powershell
python scripts/run_naive_rag_baseline.py `
  --config config/wiki18_21m_sharded_official_baseline.yaml `
  --datasets hotpotqa `
  --qa-path data/filtered/hotpotqa_all_gold_covered.jsonl `
  --use-reranker `
  --reranker-retriever-rank-weight 0.4 `
  --output-root experiments/runs/hotpot_official_baseline `
  --progress-every 25 `
  --progress-min-seconds 60
```

### 2.3 Official closure-safe E2E sanity

```powershell
python scripts/run_naive_rag_baseline.py `
  --config config/wiki18_21m_sharded_official_llm_sanity_v2.yaml `
  --datasets hotpotqa `
  --qa-path data/filtered/hotpotqa_all_gold_covered.jsonl `
  --max-queries 50 `
  --use-reranker `
  --reranker-retriever-rank-weight 0.4 `
  --continue-on-generation-error `
  --output-root experiments/runs/hotpot_closure_e2e_v2 `
  --progress-every 5 `
  --progress-min-seconds 15
```

Expected run outputs:

- `run_config.json`
- `hotpotqa/metrics.json`
- `hotpotqa/predictions.json`
- `hotpotqa/summary_metrics.json`
- `hotpotqa/progress.json`

---

## 3. Official Reference Metrics

Primary retrieval report:

- [hotpot_stage2_matrix_progress_20260308.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_stage2_matrix_progress_20260308.md)

Reference retrieval runs:

- `200 query` historical baseline:
  - [hotpot_retrieval_dense_sharded_20260308_132925](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/stage2_hotpot_matrix/hotpot_retrieval_dense_sharded_20260308_132925)
- `coverage-filtered full` historical baseline:
  - [hotpot_retrieval_dense_sharded_20260308_154210](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/stage2_hotpot_matrix/hotpot_retrieval_dense_sharded_20260308_154210)
- `200 query` closure recheck:
  - [naive_baseline_dense_sharded_20260310_095517_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_retrieval_recheck/naive_baseline_dense_sharded_20260310_095517_rerank)

Frozen retrieval reference values:

| Split | RecallAnyGoldTitle@k | RecallAllGold@k_title | RecallAllGold@raw_title | AvgRetrievalLatencyMs | AvgRerankLatencyMs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `200 query` historical | `0.72` | `0.235` | `0.24` | `2984.38` | `168.77` |
| `200 query` closure recheck | `0.72` | `0.235` | `0.245` | `475.57` | `203.56` |
| `coverage-filtered full` historical | `0.8901` | `0.3919` | `0.3919` | `1404.36` | `106.13` |

The closure recheck reproduces the official `200 query` all-gold target exactly, so the baseline is considered stable.

---

## 4. Official End-to-End Sanity Path

Historical failed attempt:

- [wiki18_21m_sharded_official_llm_sanity.yaml](D:/Project/Toy/RAG-Benchmark-Study/config/wiki18_21m_sharded_official_llm_sanity.yaml)
- result:
  - [naive_baseline_dense_sharded_20260310_001002_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_e2e/naive_baseline_dense_sharded_20260310_001002_rerank)
- outcome:
  - `GenerationErrorRate = 0.16`
  - blocker: completion budget exhaustion at `max_completion_tokens = 512`

Closure-safe official sanity config:

- [wiki18_21m_sharded_official_llm_sanity_v2.yaml](D:/Project/Toy/RAG-Benchmark-Study/config/wiki18_21m_sharded_official_llm_sanity_v2.yaml)

Frozen sanity settings:

- same retrieval backbone as official baseline
- `generation.mode = openai_compatible`
- `generator.max_completion_tokens = 1536`
- `generator.reasoning_split = true`

Closure-safe E2E reference run:

- [naive_baseline_dense_sharded_20260310_093249_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_e2e_v2/naive_baseline_dense_sharded_20260310_093249_rerank)

Reference sanity values:

| Metric | Value |
| --- | ---: |
| `EM` | `0.04` |
| `F1` | `0.1290` |
| `RecallAllGold@k_title` | `0.38` |
| `GenerationErrorRate` | `0.08` |
| `AvgGenerationLatencyMs` | `18942.53` |

This config is not the official quality baseline. It is the official closure path for proving that the best retrieval stack can be connected to the real LLM backend with acceptable generation reliability.

---

## 5. Notes

- Do not treat `IVF4096` as the official benchmark backend for this phase.
- Do not treat current `dense_sharded_title_prefilter` as a promoted retrieval method.
- Do not replace the official retrieval baseline with `hotpot_decompose`; its implementation is now stable, but it did not produce a new official winning result.
