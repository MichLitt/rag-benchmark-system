# Hotpot Closure E2E Sanity V2 Report

## 1. Goal

This report records the closure-safe end-to-end sanity reruns that were executed after the original `512`-token sanity attempt failed the `GenerationErrorRate <= 0.10` gate.

The goal was narrow:

- keep the official retrieval baseline unchanged
- repair only the generation-side completion-budget blocker
- determine whether Phase 1 can be called minimally closed

---

## 2. Runs And Escalation Path

### Attempt 1: historical sanity

Config:

- [wiki18_21m_sharded_official_llm_sanity.yaml](D:/Project/Toy/RAG-Benchmark-Study/config/wiki18_21m_sharded_official_llm_sanity.yaml)

Run:

- [naive_baseline_dense_sharded_20260310_001002_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_e2e/naive_baseline_dense_sharded_20260310_001002_rerank)

Result:

- `GenerationErrorRate = 0.16`
- `NumGenerationErrors = 8`

Observed error class:

- `8` cases:
  - `RuntimeError: LLM response exhausted the completion budget before emitting a final answer. Increase generation.max_completion_tokens.`

### Attempt 2: v2 first pass

Config basis:

- [wiki18_21m_sharded_official_llm_sanity_v2.yaml](D:/Project/Toy/RAG-Benchmark-Study/config/wiki18_21m_sharded_official_llm_sanity_v2.yaml)

Run:

- [naive_baseline_dense_sharded_20260310_091533_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_e2e_v2/naive_baseline_dense_sharded_20260310_091533_rerank)

Effective override:

- `generator.max_completion_tokens = 1024`
- `generator.reasoning_split = true`

Result:

- `GenerationErrorRate = 0.14`
- `NumGenerationErrors = 7`

Observed error class:

- `7` cases:
  - `RuntimeError: LLM response exhausted the completion budget before emitting a final answer. Increase generation.max_completion_tokens.`

Interpretation:

- reliability improved slightly
- the remaining blocker was still purely completion-budget related
- this justified one final escalation to `1536`

### Attempt 3: closure-safe final

Frozen config:

- [wiki18_21m_sharded_official_llm_sanity_v2.yaml](D:/Project/Toy/RAG-Benchmark-Study/config/wiki18_21m_sharded_official_llm_sanity_v2.yaml)

Run:

- [naive_baseline_dense_sharded_20260310_093249_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_e2e_v2/naive_baseline_dense_sharded_20260310_093249_rerank)

Frozen generation settings:

- `generator.max_completion_tokens = 1536`
- `generator.reasoning_split = true`

---

## 3. Final V2 Results

Primary metrics from [metrics.json](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_e2e_v2/naive_baseline_dense_sharded_20260310_093249_rerank/hotpotqa/metrics.json):

| Metric | Value |
| --- | ---: |
| `EM` | `0.04` |
| `F1` | `0.1290` |
| `RecallAnyGoldTitle@k` | `0.84` |
| `RecallAllGold@k_title` | `0.38` |
| `RecallAllGold@raw_title` | `0.38` |
| `AvgRetrievalLatencyMs` | `532.69` |
| `AvgRerankLatencyMs` | `203.88` |
| `AvgGenerationLatencyMs` | `18942.53` |
| `AvgLatencyMs` | `19679.09` |
| `NumGenerationErrors` | `4` |
| `GenerationErrorRate` | `0.08` |
| `TotalGenerationCostUsd` | `0` |

Generation error classes from [predictions.json](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_e2e_v2/naive_baseline_dense_sharded_20260310_093249_rerank/hotpotqa/predictions.json):

- `3` cases:
  - `RuntimeError: LLM response exhausted the completion budget before emitting a final answer. Increase generation.max_completion_tokens.`
- `1` case:
  - `TimeoutError: The read operation timed out`

Failure bucket counts:

- `both_gold_in_final = 19`
- `only_one_gold_in_raw = 24`
- `no_gold_in_raw = 7`

---

## 4. Pass / Fail Decision

Phase closure acceptance rule:

- `GenerationErrorRate <= 0.10`

Final v2 outcome:

- actual `GenerationErrorRate = 0.08`

Decision:

- **pass**

Interpretation:

- the end-to-end chain is stable enough for minimum closure
- the original blocker was mostly a completion-budget issue
- the retrieval side did not need to be reopened to achieve closure

The remaining `4/50` generation failures are acceptable under the current closure rule and should be treated as residual generation caveats, not as a reason to reopen Phase 1.
