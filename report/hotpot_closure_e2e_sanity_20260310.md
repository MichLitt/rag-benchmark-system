# Hotpot Closure E2E Sanity Report

## 1. Goal

This run was executed to provide the smallest official end-to-end sanity benchmark for the current Hotpot phase.

It was not designed to produce the best possible Hotpot answer quality. Its purpose was narrower:

- prove that the official retrieval baseline can connect to a real `openai_compatible` LLM backend
- produce one small, reproducible end-to-end run artifact
- determine whether there is any obvious generation-stage blocker before calling the phase "basically closed"

---

## 2. Frozen Setup

Frozen config:

- [wiki18_21m_sharded_official_llm_sanity.yaml](D:/Project/Toy/RAG-Benchmark-Study/config/wiki18_21m_sharded_official_llm_sanity.yaml)

Run command:

```powershell
python scripts/run_naive_rag_baseline.py `
  --config config/wiki18_21m_sharded_official_llm_sanity.yaml `
  --datasets hotpotqa `
  --qa-path data/filtered/hotpotqa_all_gold_covered.jsonl `
  --max-queries 50 `
  --use-reranker `
  --reranker-retriever-rank-weight 0.4 `
  --continue-on-generation-error `
  --output-root experiments/runs/hotpot_closure_e2e `
  --progress-every 5 `
  --progress-min-seconds 15
```

Run output:

- [naive_baseline_dense_sharded_20260310_001002_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_e2e/naive_baseline_dense_sharded_20260310_001002_rerank)

Input scope:

- dataset: `hotpotqa`
- qa path: `data/filtered/hotpotqa_all_gold_covered.jsonl`
- `max_queries = 50`

---

## 3. Results

Primary metrics from [metrics.json](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_e2e/naive_baseline_dense_sharded_20260310_001002_rerank/hotpotqa/metrics.json):

| Metric | Value |
| --- | ---: |
| `EM` | `0.02` |
| `F1` | `0.1364` |
| `RecallAnyGoldTitle@k` | `0.84` |
| `RecallAllGold@k_title` | `0.38` |
| `RecallAllGold@raw_title` | `0.38` |
| `AvgRetrievalLatencyMs` | `520.04` |
| `AvgRerankLatencyMs` | `202.32` |
| `AvgGenerationLatencyMs` | `8802.96` |
| `AvgLatencyMs` | `9525.33` |
| `NumGenerationErrors` | `8` |
| `GenerationErrorRate` | `0.16` |
| `TotalGenerationCostUsd` | `0` |

Failure bucket counts:

- `both_gold_in_final = 19`
- `only_one_gold_in_raw = 24`
- `no_gold_in_raw = 7`

Interpretation:

- the retrieval side remained coherent with the official baseline family
- the real LLM path did run end-to-end
- the system produced interpretable answer metrics, not just retrieval-only outputs

---

## 4. Generation Failure Analysis

Observed generation error categories from [predictions.json](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_e2e/naive_baseline_dense_sharded_20260310_001002_rerank/hotpotqa/predictions.json):

- `8` cases:
  - `RuntimeError: LLM response exhausted the completion budget before emitting a final answer. Increase generation.max_completion_tokens.`

This is important because it means:

- the main blocker is not provider connectivity
- the main blocker is not prompt formatting corruption
- the run is functionally alive, but the current `max_completion_tokens = 512` is too small for part of the workload

Under the closure rule defined for this phase:

- acceptable-with-caveat threshold was `GenerationErrorRate <= 0.10`
- actual result was `0.16`

So this run should be treated as:

- **end-to-end chain proven**
- **but not yet a clean pass**

---

## 5. Conclusion

This benchmark successfully proves that the project can run the full Hotpot stack with:

- official flat `21M dense_sharded` retrieval
- reranker enabled
- real `openai_compatible` generation

However, it does **not** fully satisfy the closure acceptance bar because generation errors exceeded the allowed threshold.

The blocker is narrow and concrete:

- increase or otherwise relax the generation completion budget for the official sanity route

This does not reopen the retrieval phase. It is a generation-side closure blocker.
