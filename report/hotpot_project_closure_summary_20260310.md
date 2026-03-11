# Hotpot Project Closure Summary

## 1. Executive Summary

This report consolidates the current Hotpot-focused work in this repository and evaluates how close the project is to a minimum viable closure.

The short version is:

- the project has already completed the core retrieval-side loop
- the `21M flat dense_sharded` backbone is now the trusted default backend
- the main metric is clearly fixed as `RecallAllGold@k_title`
- the remaining bottleneck is no longer corpus coverage, but second-hop title recall
- two major follow-up branches were tested and did not justify promotion:
  - `IVF4096` as the new default backend
  - `dense_sharded_title_prefilter` as a title-side quick fix
- `hotpot_decompose` now has a stable implementation, but it did not push the best `200-query` result beyond `0.24`
- the official closure-safe end-to-end sanity path now passes with `GenerationErrorRate = 0.08`
- the official `200-query` retrieval recheck reproduces `RecallAllGold@k_title = 0.235`

Minimum closure is now achieved for Phase 1.

---

## 2. What Has Been Done

### 2.1 Early retrieval and generation exploration

Before the Hotpot-specific 21M work stabilized, the project already explored:

- baseline `dense / bm25 / hybrid` retrieval
- reranker integration
- `extractive` and `openai_compatible` generation backends
- early `HyDE` and `hotpot_decompose` style query-expansion runs

These runs were useful for building the framework, but they are not the main evidence base for the final Hotpot conclusions.

### 2.2 1M -> 21M corpus upgrade

Formal report:

- [hotpot_21m_improvement_analysis_20260308.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_21m_improvement_analysis_20260308.md)

Main conclusion:

- the original low-recall problem was first a corpus-coverage problem
- upgrading to `wiki18_21m` moved the project out of the "missing evidence" regime

Key retrieval-only change:

| Metric | Early 1M baseline | 21M baseline |
| --- | ---: | ---: |
| `CoverageAny` | `0.24` | `0.94` |
| `CoverageAll` | `0.04` | `0.645` |
| `RecallAllGold@k_title` | `0.04` | `0.185` |

Meaning:

- corpus coverage was the first bottleneck
- after 21M, retrieval itself became the dominant bottleneck

### 2.3 Stage 2 flat 21M retrieval matrix

Formal report:

- [hotpot_stage2_matrix_progress_20260308.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_stage2_matrix_progress_20260308.md)

What was run:

- a full `18-config` `50-query` matrix
- best-2 reruns on `200 query`
- best rerun on `coverage-filtered full`

Sweep dimensions:

- `retrieve_top_k = 50 / 100 / 150`
- `title_first_rerank = false / true`
- `reranker_retriever_rank_weight = 0.0 / 0.2 / 0.4`

Main result:

- best `200-query` retrieval-only result:
  - `RecallAllGold@k_title = 0.235`
- best `coverage-filtered full` result:
  - `RecallAllGold@k_title = 0.3919`

Best stable stage-2 operating point:

- `retriever = dense_sharded`
- `retrieve_top_k = 100`
- `dedup_mode = title`
- `dedup_before_rerank = true`
- `use_reranker = true`
- `title_first_rerank = false`
- `reranker_retriever_rank_weight = 0.4`

What this stage proved:

- `retrieve_top_k = 100` is materially better than `50`
- `150` did not show enough extra value
- `title_first_rerank` did not become a winner
- the dominant failure bucket is still `no_gold_in_raw + only_one_gold_in_raw`

### 2.4 IVF acceleration attempt

Formal report:

- [hotpot_ivf_benchmark_20260308.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_ivf_benchmark_20260308.md)

What was done:

- implemented ANN support for `dense_sharded`
- rebuilt a full `21M ivf_flat` index
- tuned `nprobe = 16 -> 24 -> 32`

Final benchmark result:

| Split | Flat | IVF best | Decision |
| --- | ---: | ---: | --- |
| `200 query RecallAllGold@k_title` | `0.235` | `0.22` | fail |
| `AvgRetrievalLatencyMs` | `2984.38` | `25.81` | pass |

Conclusion:

- IVF solved speed
- IVF failed the recall gate
- `flat 21M dense_sharded` remains the official benchmark backend

### 2.5 Failure taxonomy and blocker analysis

Formal reports:

- [hotpot_failure_taxonomy_20260309.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_failure_taxonomy_20260309.md)
- [hotpot_failure_taxonomy_20260309_title_bm25.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_failure_taxonomy_20260309_title_bm25.md)

Two rounds were completed:

1. taxonomy on the best stage-2 flat run
2. taxonomy rerun on the stabilized `hotpot_decompose` run with full title BM25 probing

Important shift:

- before title-BM25 probing:
  - `query_formulation_gap = 96 / 200`
- after title-BM25 probing on the stabilized run:
  - `query_formulation_gap = 64 / 200`
  - `budget_limited = 45 / 200`
  - `normalization_or_alias_suspect = 39 / 200`
  - `embedding_confusion = 2 / 200`

Interpretation:

- query formulation is still the largest single blocker
- but title-side issues are collectively large enough to matter
- rerank loss is consistently small

### 2.6 Query-side decompose stabilization

Key runs:

- screening winner before stabilization:
  - [naive_baseline_dense_sharded_20260309_155449_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_query_screening/naive_baseline_dense_sharded_20260309_155449_rerank)
- stabilized screening:
  - [naive_baseline_dense_sharded_20260309_205840_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_query_screening_stability/naive_baseline_dense_sharded_20260309_205840_rerank)
- stabilized `200-query` validation:
  - [naive_baseline_dense_sharded_20260309_214927_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_query_validation_stability/naive_baseline_dense_sharded_20260309_214927_rerank)

What was changed:

- prompt tightening
- parser salvage improvements
- fallback behavior
- query-expansion failure diagnostics
- replay harness for failed decomposition samples

Most important evidence:

- replay on the historical `81` query-expansion failures produced usable queries for `81 / 81`
- `QueryExpansionErrorRate` dropped from `81 / 200 = 0.405` to `0 / 200`

But the final retrieval gain stalled:

| Config | `200-query RecallAllGold@k_title` |
| --- | ---: |
| stage-2 best flat baseline | `0.235` |
| earlier decompose winner | `0.24` |
| stabilized decompose | `0.24` |

Conclusion:

- the decompose implementation is now much more stable
- but parser stability alone did not raise the best `200-query` score beyond `0.24`

### 2.7 Title-prefilter formal screening

This branch used:

- full title BM25 manifest:
  - `E:/rag-benchmark-indexes/wiki18_21m_title_bm25/manifest.json`
- retriever:
  - `dense_sharded_title_prefilter`

Screening runs:

- `off + k=30`:
  - [naive_baseline_dense_sharded_title_prefilter_20260309_230251_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_title_prefilter_screening/naive_baseline_dense_sharded_title_prefilter_20260309_230251_rerank)
- `off + k=50`:
  - [naive_baseline_dense_sharded_title_prefilter_20260309_231504_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_title_prefilter_screening/naive_baseline_dense_sharded_title_prefilter_20260309_231504_rerank)
- `noorig3_stability + k=30`:
  - [naive_baseline_dense_sharded_title_prefilter_20260309_234005_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_title_prefilter_screening/naive_baseline_dense_sharded_title_prefilter_20260309_234005_rerank)
- `noorig3_stability + k=50`:
  - [naive_baseline_dense_sharded_title_prefilter_20260309_235120_rerank](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/hotpot_title_prefilter_screening/naive_baseline_dense_sharded_title_prefilter_20260309_235120_rerank)

Summary table:

| Protocol | Baseline | Title Prefilter Result | Decision |
| --- | ---: | ---: | --- |
| `off` | `0.00` | `k=30 -> 0.00` | fail |
| `off` | `0.00` | `k=50 -> 0.00` | fail |
| `noorig3_stability` | `0.10` | `k=30 -> 0.10` | no gain |
| `noorig3_stability` | `0.10` | `k=50 -> 0.08` | worse |

Conclusion:

- the current `dense_sharded_title_prefilter` implementation did not improve recall
- it should not be promoted to `200-query` validation

---

## 3. Current Best-Known Project State

### 3.1 Trusted default backend

The current trusted benchmark backend is:

- `flat 21M dense_sharded`

Rejected as defaults:

- `ivf_flat`
- `dense_sharded_title_prefilter`

### 3.2 Current best stable retrieval config

For formal retrieval-only benchmarking, the safest current reference config is still:

- `retriever = dense_sharded`
- `retrieve_top_k = 100`
- `dedup_mode = title`
- `dedup_before_rerank = true`
- `use_reranker = true`
- `title_first_rerank = false`
- `reranker_retriever_rank_weight = 0.4`
- `query_expansion = off`

Why this remains the stable reference:

- it produced the best confirmed `coverage-filtered full` result:
  - `RecallAllGold@k_title = 0.3919`
- it is not dependent on slower query-expansion calls
- it already has full-stage evidence, not just `50` or `200` samples

### 3.3 Best experimental candidate

The best experimental `200-query` candidate is:

- the stabilized `hotpot_decompose` route
- `RecallAllGold@k_title = 0.24`
- `QueryExpansionErrorRate = 0.0`

Why it is not yet the official default:

- it does not beat the earlier `0.24` winner
- it was not promoted to a new full `coverage-filtered` win
- its latency cost is much higher than the stable `off` route

---

## 4. What The Project Has Actually Answered

At this point the project has already answered several important questions:

1. **Was the original Hotpot failure mainly a corpus problem?**
   - yes
   - moving from the earlier corpus regime to `21M` changed the problem substantially

2. **Is flat `21M dense_sharded` good enough to serve as the main research baseline?**
   - yes
   - it is now the main trustworthy retrieval backend

3. **Should IVF replace flat as the main backend?**
   - no
   - speed gain is excellent, recall retention is insufficient

4. **Was decompose failing mainly because of parser/prompt fragility?**
   - partly yes
   - the implementation fragility was real and is now mostly fixed

5. **Does the stabilized decompose branch clearly raise the retrieval ceiling?**
   - not yet
   - it stabilized behavior without surpassing the current best `200-query` score

6. **Does the current title-prefilter implementation help?**
   - no
   - not in its present form

---

## 5. Gap To Minimum Closure

Here "minimum closure" means:

- one trustworthy default retrieval baseline
- one consolidated project summary
- one rejected acceleration path with evidence
- one diagnostic explanation of remaining failure modes
- one small but explicit end-to-end validation of the full RAG stack

Against that definition, the project status is:

### Already closed

- trustworthy default retrieval backend: **done**
- clear main metric (`RecallAllGold@k_title`): **done**
- speedup branch evaluated and rejected (`IVF4096`): **done**
- failure taxonomy and blocker explanation: **done**
- recent query-side and title-side follow-up branches evaluated: **done**
- consolidated report: **done by this report**

### Now closed

1. **Official frozen runbook**
   - [hotpot_official_baseline_runbook_20260310.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_official_baseline_runbook_20260310.md)

2. **Official closure-safe small E2E sanity benchmark**
   - historical failed attempt:
     - [hotpot_closure_e2e_sanity_20260310.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_closure_e2e_sanity_20260310.md)
   - final passing rerun:
     - [hotpot_closure_e2e_sanity_v2_20260310.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_closure_e2e_sanity_v2_20260310.md)

3. **Single formal stop-here decision memo**
   - [hotpot_phase1_stop_here_decision_20260310.md](D:/Project/Toy/RAG-Benchmark-Study/report/hotpot_phase1_stop_here_decision_20260310.md)

---

## 6. Closure Outcome

The final closure path was:

1. freeze the official retrieval baseline
2. run the original `50`-query LLM sanity attempt
3. classify generation failures
4. rerun the same `50` queries with a closure-safe completion budget
5. rerun the official `200-query` retrieval control

Final closure status:

- official retrieval baseline frozen and rechecked
- official closure-safe E2E sanity path frozen and passed
- stop-here memo written

This means the project is now closed at the intended minimum level for Phase 1, without claiming that end-to-end Hotpot quality is solved.

---

## 7. Final Assessment

The project is now **Phase-1 closed**.

The strongest stable conclusion today is:

> Use `flat 21M dense_sharded` as the official Hotpot retrieval baseline. Keep `RecallAllGold@k_title` as the main selection metric. Treat `IVF4096` and the current `dense_sharded_title_prefilter` as rejected branches for the current phase. Treat stabilized `hotpot_decompose` as a useful implementation improvement, but not yet a new winning method.

The next phase should not reopen this closure work. It should start from the frozen baseline and focus on stronger query formulation or earlier title/page-aware candidate generation.
