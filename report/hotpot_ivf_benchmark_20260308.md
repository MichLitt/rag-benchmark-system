# Hotpot IVF Benchmark Report

## 1. Status

This report tracks the `21M ivf_flat` benchmark and tuning stage that follows the existing `flat` 21M baseline.

Current status on March 9, 2026:

- benchmark code path is implemented
- preflight checks passed
- missing `coverage-filtered` subset was rebuilt
- a small-scale `ivf_flat` smoke build and `1 query` sanity eval succeeded
- the full `21M` IVF rebuild completed successfully
- `1 / 50 / 200` evaluation completed
- `nprobe = 16 -> 24 -> 32` tuning completed
- `coverage-filtered full` was not run because the `200 query` recall gate failed at all allowed `nprobe` values

This report now records the actual IVF benchmark outcome for this round.

---

## 2. Fixed Baseline

The comparison baseline remains the existing `flat` `21M dense_sharded` result set from stage 2.

Primary baseline numbers:

| Split | RecallAnyGoldTitle@k | RecallAllGold@k_title | AvgRetrievalLatencyMs | AvgRerankLatencyMs |
| --- | ---: | ---: | ---: | ---: |
| `200 query` | `0.72` | `0.235` | `2984.38` | `168.77` |
| `coverage-filtered full` | `0.8901` | `0.3919` | `1404.36` | `106.13` |

Acceptance thresholds for IVF:

- `200 query RecallAllGold@k_title >= 0.235 * 0.98 = 0.2303`
- `coverage-filtered full RecallAllGold@k_title >= 0.3919 * 0.98 = 0.3841`
- `coverage-filtered` total throughput improvement `>= 5x`

Reference baseline report:

- [hotpot_stage2_matrix_progress_20260308.md](/d:/Project/Toy/RAG-Benchmark-Study/report/hotpot_stage2_matrix_progress_20260308.md)

---

## 3. Preflight

Preflight completed locally before the full benchmark:

- GPU availability:
  - `cuda_available = True`
  - device: `NVIDIA GeForce RTX 5070 Laptop GPU`
- index disk target exists:
  - `E:\rag-benchmark-indexes\wiki18_21m_dense_sharded\manifest.json`
- source corpus exists:
  - `data/raw/corpus/wiki18_21m/passages.jsonl.gz`
- Hotpot QA exists:
  - `data/raw/flashrag/hotpotqa/dev/qa.jsonl`
- `coverage-filtered` subset rebuilt:
  - `data/filtered/hotpotqa_all_gold_covered.jsonl`
  - `total_queries = 7405`
  - `kept_queries = 4914`
  - `kept_ratio = 0.6636056718433491`

Validation commands completed successfully:

```powershell
python -m py_compile scripts/build_dense_sharded_index.py scripts/eval_hotpot_retrieval.py
python -m pytest tests/test_build_dense_sharded_index.py tests/test_sharded_dense.py tests/test_hotpot_retrieval_eval.py -q
```

---

## 4. Smoke Validation

A small `ivf_flat` smoke index was built and queried successfully to validate the new CLI and manifest behavior.

Smoke build command:

```powershell
python scripts/build_dense_sharded_index.py `
  --corpus-path data/raw/corpus/wiki18_21m/passages.jsonl.gz `
  --output-root data/indexes/wiki18_21m_ivf_smoke_v2 `
  --index-type ivf_flat `
  --ivf-nlist 256 `
  --nprobe 16 `
  --batch-size 64 `
  --device cuda `
  --max-docs 20000 `
  --shard-size 10000
```

Smoke manifest checks:

- root `index_type = ivf_flat`
- root `nprobe = 16`
- shard `nlist = [200, 200]`

Smoke sanity eval command:

```powershell
python scripts/eval_hotpot_retrieval.py `
  --retriever dense_sharded `
  --manifest data/indexes/wiki18_21m_ivf_smoke_v2/manifest.json `
  --top-k 20 `
  --retrieve-top-k 100 `
  --dedup-mode title `
  --dedup-before-rerank `
  --use-reranker `
  --reranker-retriever-rank-weight 0.4 `
  --batch-size 8 `
  --nprobe 16 `
  --coverage-cache-path data/indexes/wiki18_21m_ivf_smoke_v2/titles_cache.json.gz `
  --refresh-coverage-cache `
  --max-queries 1 `
  --output-dir experiments/runs/stage3_hotpot_ivf_benchmark
```

Smoke sanity output:

- output dir:
  - `experiments/runs/stage3_hotpot_ivf_benchmark/hotpot_retrieval_dense_sharded_20260308_180146`
- required runtime fields present:
  - `RetrievalMode = ivf_flat`
  - `NProbe = 16`
  - `NumWorkers = 2`
  - `CoverageCachePath = data\indexes\wiki18_21m_ivf_smoke_v2\titles_cache.json.gz`

This smoke run is only a wiring check. Its retrieval metric values are not used as benchmark conclusions because the corpus is truncated to `20,000` docs.

---

## 5. Full 21M Execution Commands

### 5.1 Build The Full IVF Index

```powershell
python scripts/build_dense_sharded_index.py `
  --corpus-path data/raw/corpus/wiki18_21m/passages.jsonl.gz `
  --output-root E:/rag-benchmark-indexes/wiki18_21m_dense_sharded_ivf4096_np16 `
  --index-type ivf_flat `
  --ivf-nlist 4096 `
  --nprobe 16 `
  --batch-size 64 `
  --device cuda
```

Post-build checks:

- root `index_type = ivf_flat`
- root `nprobe = 16`
- each shard has non-null `index_type`
- each shard has non-null `nlist`
- actual output:
  - manifest path: `E:\rag-benchmark-indexes\wiki18_21m_dense_sharded_ivf4096_np16\manifest.json`
  - `total_docs = 21,015,324`
  - `num_workers = 22`

Operational note:

- on this machine, `uv run python` uses CPU-only `torch`
- the actual full build used system `python`
- resume support was added and used after an interrupted run:
  - `python scripts/build_dense_sharded_index.py ... --resume`

### 5.2 Run The 4 Benchmark Stages

Common flags:

```powershell
--retriever dense_sharded `
--manifest E:/rag-benchmark-indexes/wiki18_21m_dense_sharded_ivf4096_np16/manifest.json `
--top-k 20 `
--retrieve-top-k 100 `
--dedup-mode title `
--dedup-before-rerank `
--use-reranker `
--reranker-retriever-rank-weight 0.4 `
--batch-size 32 `
--nprobe 16 `
--coverage-cache-path E:/rag-benchmark-indexes/wiki18_21m_dense_sharded_ivf4096_np16/titles_cache.json.gz `
--output-dir experiments/runs/stage3_hotpot_ivf_benchmark
```

Sanity:

```powershell
python scripts/eval_hotpot_retrieval.py `
  --retriever dense_sharded `
  --manifest E:/rag-benchmark-indexes/wiki18_21m_dense_sharded_ivf4096_np16/manifest.json `
  --top-k 20 `
  --retrieve-top-k 100 `
  --dedup-mode title `
  --dedup-before-rerank `
  --use-reranker `
  --reranker-retriever-rank-weight 0.4 `
  --batch-size 32 `
  --nprobe 16 `
  --coverage-cache-path E:/rag-benchmark-indexes/wiki18_21m_dense_sharded_ivf4096_np16/titles_cache.json.gz `
  --refresh-coverage-cache `
  --max-queries 1 `
  --output-dir experiments/runs/stage3_hotpot_ivf_benchmark
```

`50 query` and `200 query`:

- same command, only change `--max-queries 50`
- then `--max-queries 200`

`coverage-filtered full`:

```powershell
python scripts/eval_hotpot_retrieval.py `
  --qa-path data/filtered/hotpotqa_all_gold_covered.jsonl `
  --retriever dense_sharded `
  --manifest E:/rag-benchmark-indexes/wiki18_21m_dense_sharded_ivf4096_np16/manifest.json `
  --top-k 20 `
  --retrieve-top-k 100 `
  --dedup-mode title `
  --dedup-before-rerank `
  --use-reranker `
  --reranker-retriever-rank-weight 0.4 `
  --batch-size 32 `
  --nprobe 16 `
  --coverage-cache-path E:/rag-benchmark-indexes/wiki18_21m_dense_sharded_ivf4096_np16/titles_cache.json.gz `
  --output-dir experiments/runs/stage3_hotpot_ivf_benchmark
```

---

## 6. Tuning Log

Only `nprobe` may be changed in this round.

| Attempt | nprobe | 200 RecallAllGold@k_title | coverage-filtered RecallAllGold@k_title | Recall Constraint | Speed Constraint | Decision |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | `16` | `0.205` | `not run` | fail (`< 0.2303`) | pass on retrieval latency (`25.93 ms`) | escalate to `24` |
| 2 | `24` | `0.215` | `not run` | fail (`< 0.2303`) | pass on retrieval latency (`24.90 ms`) | escalate to `32` |
| 3 | `32` | `0.22` | `not run` | fail (`< 0.2303`) | pass on retrieval latency (`25.81 ms`) | stop, IVF4096 rejected |

Rules:

1. Start with `nprobe = 16`
2. If recall fails on `200 query` or `coverage-filtered full`, move to `24`
3. If recall still fails, move to `32`
4. If `32` still fails, conclude `IVF4096` is not acceptable

Observed run outputs:

- `1 query sanity`
  - run dir: `experiments/runs/stage3_hotpot_ivf_benchmark/hotpot_retrieval_dense_sharded_20260309_142441`
  - `RetrievalMode = ivf_flat`
  - `NProbe = 16`
  - `NumWorkers = 22`
  - `CoverageCachePath` present
- `50 query screening`
  - run dir: `experiments/runs/stage3_hotpot_ivf_benchmark/hotpot_retrieval_dense_sharded_20260309_142536`
  - `RecallAllGold@k_title = 0.26`
  - `AvgRetrievalLatencyMs = 32.67`
- `200 query validation`
  - `nprobe = 16`
    - run dir: `experiments/runs/stage3_hotpot_ivf_benchmark/hotpot_retrieval_dense_sharded_20260309_142651`
  - `nprobe = 24`
    - run dir: `experiments/runs/stage3_hotpot_ivf_benchmark/hotpot_retrieval_dense_sharded_20260309_142801`
  - `nprobe = 32`
    - run dir: `experiments/runs/stage3_hotpot_ivf_benchmark/hotpot_retrieval_dense_sharded_20260309_142906`

---

## 7. Final Comparison Table

| Split | Flat RecallAllGold@k_title | IVF RecallAllGold@k_title | Relative Recall Retention | Flat AvgRetrievalLatencyMs | IVF AvgRetrievalLatencyMs | Throughput Gain | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `200 query` | `0.235` | `0.22` | `93.6%` | `2984.38` | `25.81` | `115.6x` | recall fail |
| `coverage-filtered full` | `0.3919` | `not run` | `not run` | `1404.36` | `not run` | `not run` | skipped after 200 fail |

Final go/no-go rule:

- if recall retention is `>= 98%` and throughput gain is `>= 5x`, promote IVF to the new default benchmark backend
- if recall passes but speed does not, keep IVF as a candidate backend
- if recall fails, retain flat as the default backend

Current conclusion:

- `IVF4096` is not acceptable as the default Hotpot benchmark backend for this round
- the main failure reason is recall: best `200 query RecallAllGold@k_title = 0.22`, below the required `0.2303`
- speed is excellent, but speed does not compensate for recall regression under the current acceptance rule
- retain `flat` `21M dense_sharded` as the default benchmark backend
