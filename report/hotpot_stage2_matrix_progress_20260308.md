# Hotpot Stage 2 Matrix Progress Report

## Goal

This stage focuses on validating the `21M dense_sharded` retrieval setup on `HotpotQA` and selecting the best low-cost retrieval configuration before deciding whether any further method change is justified.

The decision rule for this stage was fixed in advance:

- first run an `18`-configuration `50 query` screening matrix
- then rerun the best `2` configurations on `200 query`
- then rerun the best `2` configurations on the `coverage-filtered` full subset

The main target metric is `RecallAllGold@k_title`.

## Starting Point

Before this stage, the project had already completed the following:

- built the full `wiki18_21m` dense sharded index
- added `dense_sharded` retrieval support
- added `title-first` candidate shaping
- added richer Hotpot diagnostics:
  - `gold_title_ranks`
  - `gold_titles_in_raw_candidates`
  - `gold_titles_after_dedup`
  - `gold_titles_in_final_top_k`
  - `missing_gold_count`
  - `retrieval_failure_bucket`

The previously established `21M` retrieval-only baselines were:

- full `200 query` baseline:
  - `CoverageAny = 0.94`
  - `CoverageAll = 0.645`
  - `RecallAnyGoldTitle@k = 0.65`
  - `RecallAllGold@k_title = 0.185`
- `coverage-filtered` baseline:
  - `RecallAnyGoldTitle@k = 0.8488`
  - `RecallAllGold@k_title = 0.3128`

These results already showed that corpus coverage was largely solved and that the main remaining problem had moved to multi-hop retrieval itself.

## What Was Executed

### 1. Sanity Check

A `1 query` sanity run was executed first to verify that:

- the `21M` manifest loads correctly
- the reranker initializes correctly
- the new retrieval details fields are written correctly

Sanity run output:

- [hotpot_retrieval_dense_sharded_20260308_120821](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/stage2_hotpot_matrix/hotpot_retrieval_dense_sharded_20260308_120821)

### 2. 50-Query Screening Matrix

The full `18`-configuration matrix was completed under:

- [stage2_hotpot_matrix](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/stage2_hotpot_matrix)

Sweep dimensions:

- `retrieve_top_k`: `50 / 100 / 150`
- `title_first_rerank`: `false / true`
- `reranker_retriever_rank_weight`: `0.0 / 0.2 / 0.4`

Other settings were fixed:

- `top_k = 20`
- `dedup_mode = title`
- `dedup_before_rerank = true`
- `use_reranker = true`
- `title_pool_k = 40`
- `max_chunks_per_title = 2`
- `min_unique_titles = 6`
- `reranker_model = cross-encoder/ms-marco-MiniLM-L-6-v2`
- `reranker_rank_fusion_k = 60`

### 3. Best-2 Rerun on 200 Query

The top `2` configurations from the `50 query` matrix were rerun on `200 query`.

Selected configurations:

1. `retrieve_top_k = 100`, `title_first_rerank = false`, `reranker_retriever_rank_weight = 0.0`
2. `retrieve_top_k = 100`, `title_first_rerank = false`, `reranker_retriever_rank_weight = 0.4`

Outputs:

- [hotpot_retrieval_dense_sharded_20260308_133458](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/stage2_hotpot_matrix/hotpot_retrieval_dense_sharded_20260308_133458)
- [hotpot_retrieval_dense_sharded_20260308_132925](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/stage2_hotpot_matrix/hotpot_retrieval_dense_sharded_20260308_132925)

### 4. Coverage-Filtered Full Rerun

The first `coverage-filtered` full rerun was completed with:

- `retrieve_top_k = 100`
- `title_first_rerank = false`
- `reranker_retriever_rank_weight = 0.4`

Output:

- [hotpot_retrieval_dense_sharded_20260308_154210](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/stage2_hotpot_matrix/hotpot_retrieval_dense_sharded_20260308_154210)

The second planned coverage-filtered rerun was intentionally cancelled after the first run completed, in order to save time.

Logs:

- [filtered_20260308_133540.stdout.log](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/stage2_hotpot_matrix/logs/filtered_20260308_133540.stdout.log)
- [filtered_20260308_133540.stderr.log](D:/Project/Toy/RAG-Benchmark-Study/experiments/runs/stage2_hotpot_matrix/logs/filtered_20260308_133540.stderr.log)

## Results

### 50-Query Matrix Summary

The best `50 query` result reached:

- `RecallAllGold@k_title = 0.26`
- `RecallAnyGoldTitle@k = 0.74`

Main observations from the full matrix:

- increasing `retrieve_top_k` from `50` to `100` improved `RecallAllGold@k_title` from about `0.24` to `0.26`
- increasing `retrieve_top_k` from `100` to `150` did not produce a further clear gain
- `title_first_rerank = true` did not show a stable advantage in this matrix
- changing `reranker_retriever_rank_weight` did not materially change the best retrieval quality

This means the first strong gain came from deeper raw retrieval, not from the title-first rerank path.

### 200-Query Rerun

Comparison against the earlier `21M` baseline:

| Config | RecallAnyGoldTitle@k | RecallAllGold@k_title | AvgRetrievalLatencyMs | AvgRerankLatencyMs |
| --- | ---: | ---: | ---: | ---: |
| Old 21M baseline | 0.65 | 0.185 | not directly comparable | not directly comparable |
| `retrieve_top_k=100`, `title_first=false`, `weight=0.0` | 0.71 | 0.235 | 4686.01 | 132.72 |
| `retrieve_top_k=100`, `title_first=false`, `weight=0.4` | 0.72 | 0.235 | 2984.38 | 168.77 |

Failure bucket counts for both best `200 query` runs were effectively the same:

- `only_one_gold_in_raw = 96`
- `no_gold_in_raw = 55`
- `both_gold_after_dedup_but_lost_after_rerank = 2`
- `both_gold_in_final = 47`

### Coverage-Filtered Full Rerun

The completed `coverage-filtered` run produced:

| Config | RecallAnyGoldTitle@k | RecallAllGold@k_title | AvgRetrievalLatencyMs | AvgRerankLatencyMs |
| --- | ---: | ---: | ---: | ---: |
| Old coverage-filtered baseline | 0.8488 | 0.3128 | not directly comparable | not directly comparable |
| `retrieve_top_k=100`, `title_first=false`, `weight=0.4` | 0.8901 | 0.3919 | 1404.36 | 106.13 |

Failure bucket counts:

- `only_one_gold_in_raw = 2352`
- `no_gold_in_raw = 486`
- `both_gold_after_dedup_but_lost_after_rerank = 150`
- `both_gold_in_final = 1926`

## Analysis

### What Improved

This stage did produce a real improvement over the earlier `21M` retrieval baseline:

- `RecallAnyGoldTitle@k` improved from `0.65` to `0.72`
- `RecallAllGold@k_title` improved from `0.185` to `0.235`

On the `coverage-filtered` full subset, the improvement was also clear:

- `RecallAnyGoldTitle@k` improved from `0.8488` to `0.8901`
- `RecallAllGold@k_title` improved from `0.3128` to `0.3919`

So the new validation matrix was useful: the system is better than the earlier default, and `retrieve_top_k = 100` is a better operating point than the old shallower setup.

### What Did Not Work

Two planned ideas did not show the expected value in this validation stage:

- `title_first_rerank` did not beat the simpler reranking path
- changing the reranker retrieval-rank blending weight did not materially improve the main metric

This means the current bottleneck is not primarily in reranker-side candidate shaping.

### Where the Real Bottleneck Is

The failure buckets make the next step clear.

For the best `200 query` runs:

- `151 / 200` failures came from `no_gold_in_raw` or `only_one_gold_in_raw`
- only `2 / 200` cases had `both_gold_after_dedup_but_lost_after_rerank`

So the dominant failure mode is still raw candidate recall, not rerank-stage loss.

In other words:

- the second supporting page still fails to enter the raw candidate pool often enough
- the reranker is not the main reason that both gold pages disappear

On the `coverage-filtered` full subset, the picture improves but does not fundamentally change:

- the raw retrieval side is still the dominant bottleneck because `only_one_gold_in_raw + no_gold_in_raw = 2838`
- rerank-stage loss is more visible than in the `200 query` run, but still secondary compared with raw recall loss

### Decision Against the Predefined Threshold

The predefined go/no-go rule for moving to LLM generation was:

- `RecallAllGold@k_title >= 0.28` on `200 query`
- `RecallAllGold@k_title >= 0.45` on `coverage-filtered`

At this point, the best completed `200 query` result is:

- `RecallAllGold@k_title = 0.235`

And the best completed `coverage-filtered` result is:

- `RecallAllGold@k_title = 0.3919`

So the project should **not** move into the LLM baseline stage yet.

## Recommended Next Step

Based on the planned decision rules and the observed failure buckets, the next method change should be:

1. keep the current best retrieval backbone:
   - `retrieve_top_k = 100`
   - `dedup_mode = title`
   - `dedup_before_rerank = true`
   - `use_reranker = true`
   - `reranker_retriever_rank_weight = 0.4`
2. introduce `hotpot_decompose` as the next targeted experiment
3. compare `query_expansion = off` vs `hotpot_decompose` on:
   - `50 query` first
   - then `200 query`
4. continue to use `RecallAllGold@k_title` as the main selection metric

The reason is straightforward:

- rerank-stage tuning has already shown limited upside
- raw retrieval is still missing too many second-hop gold titles
- `hotpot_decompose` is the first change that directly targets raw candidate recall

## Current Status

Completed:

- `1 query` sanity check
- full `18`-run `50 query` matrix
- best-`2` rerun on `200 query`
- first best-config rerun on the `coverage-filtered` full subset
- result analysis and stage conclusion

Current decision:

- do not move to LLM generation yet
- then start the next retrieval experiment with `hotpot_decompose`
