# Hotpot Phase 1 Stop-Here Decision

## Decision

Phase 1 should be considered **complete**, and the project should be treated as having reached **minimum practical closure** for this phase.

The official retrieval baseline for this phase is:

- [wiki18_21m_sharded_official_baseline.yaml](D:/Project/Toy/RAG-Benchmark-Study/config/wiki18_21m_sharded_official_baseline.yaml)

The official closure-safe end-to-end sanity config is:

- [wiki18_21m_sharded_official_llm_sanity_v2.yaml](D:/Project/Toy/RAG-Benchmark-Study/config/wiki18_21m_sharded_official_llm_sanity_v2.yaml)

Supporting runbook:

- [hotpot_official_baseline_runbook_20260310.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_official_baseline_runbook_20260310.md)

Project summary:

- [hotpot_project_closure_summary_20260310.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_project_closure_summary_20260310.md)

---

## Official Baseline

Frozen official baseline:

- `flat 21M dense_sharded`
- `retrieve_top_k = 100`
- `top_k = 20`
- `dedup_mode = title`
- `dedup_before_rerank = true`
- `use_reranker = true`
- `title_first_rerank = false`
- `reranker_retriever_rank_weight = 0.4`
- `query_expansion = off`

Official reference retrieval numbers:

- `200 query RecallAllGold@k_title = 0.235`
- `coverage-filtered full RecallAllGold@k_title = 0.3919`

Closure recheck run:

- [naive_baseline_dense_sharded_20260310_095517_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_retrieval_recheck/naive_baseline_dense_sharded_20260310_095517_rerank)

Closure recheck result:

- `RecallAllGold@k_title = 0.235`

So the official retrieval baseline remained stable during closure.

---

## Branch Decisions

### Rejected branches

`IVF4096`

- retrieval latency improved dramatically
- recall retention failed the acceptance gate
- decision: reject as the official benchmark backend for this phase

Current `dense_sharded_title_prefilter`

- no recall improvement on the `off` route
- no recall improvement on the `noorig3_stability` route
- one setting was worse than baseline
- decision: reject as a promoted retrieval method for this phase

### Stabilized but not promoted

`hotpot_decompose`

- parser/prompt/fallback stability is much better
- query-expansion error rate dropped from the earlier brittle regime to `0`
- best stable `200-query` result remains `0.24`
- it did not become a new official winner

Decision:

- keep as a useful implementation improvement and future research branch
- do not freeze it as the official baseline

---

## Remaining Bottleneck

The remaining retrieval bottleneck is still second-hop title recall, not rerank-stage collapse.

Most important blocker family from the latest taxonomy:

- `query_formulation_gap`
- `budget_limited`
- `normalization_or_alias_suspect`

Rerank-stage loss is consistently small compared with raw candidate miss.

This means future work should not center on:

- more parser-stability tuning
- more current title-prefilter variants
- more reranker-side reshaping alone

It should instead center on:

- stronger query formulation
- or earlier title/page-aware candidate generation

---

## End-to-End Closure Status

Historical sanity attempt:

- [hotpot_closure_e2e_sanity_20260310.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_closure_e2e_sanity_20260310.md)

Historical result:

- `GenerationErrorRate = 0.16`
- failure source:
  - completion budget exhaustion before final answer emission

Closure-safe sanity rerun:

- [hotpot_closure_e2e_sanity_v2_20260310.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_closure_e2e_sanity_v2_20260310.md)

Closure-safe run:

- [naive_baseline_dense_sharded_20260310_093249_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_closure_e2e_v2/naive_baseline_dense_sharded_20260310_093249_rerank)

Closure-safe result:

- `GenerationErrorRate = 0.08`
- `RecallAllGold@k_title = 0.38`

Therefore:

- the real LLM chain has been proven with an acceptable error rate
- the minimum closure gate is satisfied
- no retrieval-side reopening is required for this phase

---

## Stop-Here Recommendation

This phase should stop here.

Phase 1 is now closed with:

1. a frozen official retrieval baseline
2. a closure-safe official E2E sanity path
3. a verified `200-query` retrieval recheck

The next phase should not continue sweeping the current retrieval variants from this branch. Instead it should focus on:

1. stronger query formulation
2. earlier title/page-aware candidate generation
3. generation-side quality improvements only if they belong to the next phase's goals

Final decision:

- **Phase 1 complete; minimal closure achieved**
