# Phase 4 Experiment Findings Summary

## Setup
- 5 retrieval configs × 3 datasets × 200 queries = 3,000 total queries
- Datasets: HotpotQA (multi-hop), NQ (single-hop factoid), TriviaQA (trivia)
- Metrics: EM, F1, Recall@k, AvgFaithfulness (C1/C2 only — C3-C5 predictions lack retrieved_texts field)

## Results Table

| Config | hotpotqa F1 | nq F1 | triviaqa F1 | Avg F1 | Cost/run |
|--------|------------|-------|------------|--------|----------|
| C1 dense only | 0.101 | 0.116 | 0.176 | 0.131 | $0.35 |
| C2 +reranker | 0.131 | 0.112 | **0.212** | 0.152 | $0.34 |
| C3 +reranker topk5 | 0.138 | 0.113 | 0.187 | 0.146 | **$0.16** |
| C4 HyDE | **0.145** | 0.108 | 0.194 | 0.149 | $0.34 |
| C5 decompose | 0.123 | **0.126** | 0.190 | 0.146 | $0.35 |

## Key Findings

### Finding 1: Reranker consistently helps on hotpotqa and triviaqa
C2 vs C1: hotpotqa +30% F1 (0.101→0.131), triviaqa +20% (0.176→0.212).
NQ shows no gain (0.116→0.112), suggesting reranker is most valuable for complex multi-hop and longer-form queries.

### Finding 2: HyDE is the best single technique for hotpotqa
C4 achieves F1=0.145 on hotpotqa, the highest across all configs.
The hypothetical document helps bridge the query-document lexical gap in multi-hop questions where the query phrasing is far from the answer passage.

### Finding 3: Narrow context (topk5) achieves near-C2 quality at half the cost
C3 (top-k=5) vs C2 (top-k=20): hotpotqa 0.138 vs 0.131, triviaqa 0.187 vs 0.212.
Cost drops from ~$0.34 to ~$0.16 per run — a 53% reduction for <5% F1 degradation on most datasets.
Exception: triviaqa drops more (0.212→0.187) suggesting it benefits from wider context.

### Finding 4: Query decomposition helps on NQ but not hotpotqa
C5 achieves best NQ F1 (0.126), beating all other configs.
Surprisingly, decomposition does NOT help on hotpotqa (0.123 vs C2's 0.131).
Possible reason: the auto-decomposition used generic HyDE for sub-queries rather than true multi-hop decomposition.

### Finding 5: Faithfulness scores are low across the board (C1/C2 only)
AvgFaithfulness: 0.03–0.12, HallucinationRate: 0.88–0.97.
The LLM frequently generates answers not strictly grounded in the retrieved passages,
often hedging or using parametric knowledge. This is a known issue with smaller/cheaper models on RAG tasks.

## Retrieval vs Generation Error Analysis
High Recall@k (0.63–0.81) but low F1 (0.10–0.21) indicates the bottleneck is primarily **generation**, not retrieval.
The model retrieves relevant passages but fails to extract precise answers, especially for multi-hop reasoning.

## Recommendation for Portfolio
Best config overall: **C4 (HyDE)** for quality, **C3 (topk5)** for cost-efficiency.
