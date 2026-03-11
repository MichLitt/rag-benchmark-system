# Failure Mode Comparison — Phase 4 Matrix

### hotpotqa

| Failure Mode | C1_dense_only | C2_dense_rerank | C3_dense_rerank_topk5 | C4_rerank_hyde | C5_rerank_decompose |
| --- | ---: | ---: | ---: | ---: | ---: |
| correct | 7.5% (15) | 12.0% (24) | 13.5% (27) | 13.5% (27) | 10.5% (21) |
| no_gold_in_raw | 26.5% (53) | 26.0% (52) | 25.5% (51) | 24.5% (49) | 26.5% (53) |
| only_one_gold_in_raw | 45.5% (91) | 42.5% (85) | 41.5% (83) | 41.0% (82) | 42.5% (85) |
| both_gold_in_raw_but_lost_after_dedup | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| both_gold_after_dedup_but_lost_after_rerank | 5.0% (10) | 1.0% (2) | 6.0% (12) | 0.5% (1) | 0.5% (1) |
| both_gold_in_final | 15.5% (31) | 18.5% (37) | 13.5% (27) | 20.5% (41) | 20.0% (40) |
| generation_failure | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) |

### nq

| Failure Mode | C1_dense_only | C2_dense_rerank | C3_dense_rerank_topk5 | C4_rerank_hyde | C5_rerank_decompose |
| --- | ---: | ---: | ---: | ---: | ---: |
| correct | 6.0% (12) | 5.0% (10) | 6.0% (12) | 6.0% (12) | 8.0% (16) |
| retrieval_failure | 94.0% (188) | 95.0% (190) | 94.0% (188) | 94.0% (188) | 92.0% (184) |
| generation_failure | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) |

### triviaqa

| Failure Mode | C1_dense_only | C2_dense_rerank | C3_dense_rerank_topk5 | C4_rerank_hyde | C5_rerank_decompose |
| --- | ---: | ---: | ---: | ---: | ---: |
| correct | 15.0% (30) | 23.5% (47) | 20.0% (40) | 17.5% (35) | 19.0% (38) |
| retrieval_failure | 85.0% (170) | 76.5% (153) | 80.0% (160) | 82.5% (165) | 81.0% (162) |
| generation_failure | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) |

## Flip Cases (144 queries)

Queries where at least one config succeeded and another failed:

- **hotpotqa** `dev_104`: correct in ['C1_dense_only', 'C4_rerank_hyde'], failed in ['C2_dense_rerank', 'C3_dense_rerank_topk5', 'C5_rerank_decompose']
- **hotpotqa** `dev_105`: correct in ['C2_dense_rerank', 'C4_rerank_hyde'], failed in ['C1_dense_only', 'C3_dense_rerank_topk5', 'C5_rerank_decompose']
- **hotpotqa** `dev_11`: correct in ['C4_rerank_hyde'], failed in ['C1_dense_only', 'C2_dense_rerank', 'C3_dense_rerank_topk5', 'C5_rerank_decompose']
- **hotpotqa** `dev_110`: correct in ['C4_rerank_hyde'], failed in ['C1_dense_only', 'C2_dense_rerank', 'C3_dense_rerank_topk5', 'C5_rerank_decompose']
- **hotpotqa** `dev_123`: correct in ['C2_dense_rerank', 'C4_rerank_hyde'], failed in ['C1_dense_only', 'C3_dense_rerank_topk5', 'C5_rerank_decompose']
- **hotpotqa** `dev_126`: correct in ['C3_dense_rerank_topk5'], failed in ['C1_dense_only', 'C2_dense_rerank', 'C4_rerank_hyde', 'C5_rerank_decompose']
- **hotpotqa** `dev_130`: correct in ['C1_dense_only', 'C3_dense_rerank_topk5'], failed in ['C2_dense_rerank', 'C4_rerank_hyde', 'C5_rerank_decompose']
- **hotpotqa** `dev_132`: correct in ['C3_dense_rerank_topk5', 'C5_rerank_decompose'], failed in ['C1_dense_only', 'C2_dense_rerank', 'C4_rerank_hyde']
- **hotpotqa** `dev_136`: correct in ['C3_dense_rerank_topk5', 'C4_rerank_hyde'], failed in ['C1_dense_only', 'C2_dense_rerank', 'C5_rerank_decompose']
- **hotpotqa** `dev_14`: correct in ['C1_dense_only', 'C3_dense_rerank_topk5', 'C4_rerank_hyde', 'C5_rerank_decompose'], failed in ['C2_dense_rerank']
- **hotpotqa** `dev_151`: correct in ['C3_dense_rerank_topk5', 'C4_rerank_hyde'], failed in ['C1_dense_only', 'C2_dense_rerank', 'C5_rerank_decompose']
- **hotpotqa** `dev_155`: correct in ['C5_rerank_decompose'], failed in ['C1_dense_only', 'C2_dense_rerank', 'C3_dense_rerank_topk5', 'C4_rerank_hyde']
- **hotpotqa** `dev_161`: correct in ['C4_rerank_hyde', 'C5_rerank_decompose'], failed in ['C1_dense_only', 'C2_dense_rerank', 'C3_dense_rerank_topk5']
- **hotpotqa** `dev_163`: correct in ['C1_dense_only'], failed in ['C2_dense_rerank', 'C3_dense_rerank_topk5', 'C4_rerank_hyde', 'C5_rerank_decompose']
- **hotpotqa** `dev_168`: correct in ['C1_dense_only', 'C2_dense_rerank', 'C3_dense_rerank_topk5', 'C5_rerank_decompose'], failed in ['C4_rerank_hyde']
- **hotpotqa** `dev_170`: correct in ['C3_dense_rerank_topk5', 'C4_rerank_hyde', 'C5_rerank_decompose'], failed in ['C1_dense_only', 'C2_dense_rerank']
- **hotpotqa** `dev_173`: correct in ['C2_dense_rerank', 'C3_dense_rerank_topk5'], failed in ['C1_dense_only', 'C4_rerank_hyde', 'C5_rerank_decompose']
- **hotpotqa** `dev_179`: correct in ['C1_dense_only', 'C2_dense_rerank', 'C3_dense_rerank_topk5'], failed in ['C4_rerank_hyde', 'C5_rerank_decompose']
- **hotpotqa** `dev_18`: correct in ['C4_rerank_hyde'], failed in ['C1_dense_only', 'C2_dense_rerank', 'C3_dense_rerank_topk5', 'C5_rerank_decompose']
- **hotpotqa** `dev_182`: correct in ['C1_dense_only', 'C2_dense_rerank', 'C4_rerank_hyde'], failed in ['C3_dense_rerank_topk5', 'C5_rerank_decompose']

... and 124 more.
