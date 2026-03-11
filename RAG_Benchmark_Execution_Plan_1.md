# RAG Benchmark Study — 详细执行计划

## 项目定位

**Modular RAG System with Systematic Component-Level Evaluation on Standard Benchmarks**

核心卖点：不是做一个应用，而是系统性研究RAG架构设计，用标准benchmark量化每个组件的贡献，并从accuracy/latency/cost三个维度做工程权衡分析。

---

## 总体时间线：12-14天

| 阶段 | 天数 | 内容 |
|------|------|------|
| Phase 1 | Day 1-2 | 环境搭建 + 数据准备 + Naive RAG baseline |
| Phase 2 | Day 3-5 | Hybrid Search + Reranker |
| Phase 3 | Day 6-7 | Query Expansion (HyDE) |
| Phase 4 | Day 8-9 | Evaluation框架 + 全量实验 |
| Phase 5 | Day 10-11 | Error Analysis + Failure Mode + Cost |
| Phase 6 | Day 12-13 | 前端 + 文档 + 包装 |
| (Optional) | Day 14 | Iterative Multi-hop / 简化Graph |

---

## Phase 1: 环境搭建 + Naive RAG Baseline（Day 1-2）

### Day 1: 数据和基础设施

**上午：项目结构搭建**
```
rag-benchmark/
├── config/           # YAML配置文件
├── src/
│   ├── retrieval/    # 各种retriever实现
│   ├── reranking/    # reranker模块
│   ├── generation/   # LLM generation
│   ├── query/        # query expansion模块
│   ├── evaluation/   # 评估指标
│   └── analysis/     # error analysis工具
├── data/             # 数据集
├── experiments/      # 实验结果
├── notebooks/        # 分析notebook
└── README.md
```

**下午：数据集准备**
- 从FlashRAG/HuggingFace下载预处理数据集：
  - HotpotQA（multi-hop，~7K test questions）
  - Natural Questions（single-hop，~3.6K test）
  - TriviaQA（长尾知识，~11K test）
- 下载对应的Wikipedia corpus（FlashRAG提供预处理版本）
- 建FAISS索引：用SBERT (all-MiniLM-L6-v2) 对corpus做embedding
- 建BM25索引：用rank_bm25或Pyserini

**关键产出：** 3个数据集可用，FAISS + BM25索引构建完成

### Day 2: Naive RAG Baseline

**上午：实现基础pipeline**
- Dense retriever：FAISS相似度搜索，top-k=5
- Generator：调用LLM API（Claude/GPT），设计基础prompt
- 实现评估脚本：EM (Exact Match), F1, Recall@5
- 添加计时器：记录每个环节的latency

**下午：跑baseline + 记录结果**
- 在3个数据集上跑Naive RAG
- 记录：EM, F1, Recall@5, 平均latency, token消耗
- 保存所有中间结果（retrieved docs, generated answers）供后续error analysis
- 初步观察：哪些query答错了？

**关键产出：** Naive RAG baseline数字 + 所有中间结果存档

---

## Phase 2: Hybrid Search + Reranker（Day 3-5）

### Day 3: Hybrid Search实现

**上午：BM25 + Dense融合**
- 实现Reciprocal Rank Fusion (RRF)：
  ```
  RRF_score(d) = Σ 1/(k + rank_i(d))  # k=60 is standard
  ```
- 支持可调权重：alpha * dense_score + (1-alpha) * sparse_score
- 实现为可替换模块，和Naive RAG共享同一个evaluation pipeline

**下午：跑实验 + 调参**
- alpha从0到1，步长0.1，找最优融合比例
- 在3个数据集上跑Hybrid RAG
- 记录同样的指标：EM, F1, Recall@5, latency
- 对比Naive vs Hybrid的retrieval质量差异

**关键产出：** Hybrid search结果 + alpha敏感度分析

### Day 4: Cross-Encoder Reranker

**上午：实现reranker模块**
- 用cross-encoder/ms-marco-MiniLM-L-6-v2
- Pipeline：retriever top-20 → reranker重排 → top-5送入LLM
- 实现为独立模块，可以接在任何retriever后面

**下午：实验**
- 4种组合：
  - Dense only (baseline)
  - Dense + Reranker
  - Hybrid only
  - Hybrid + Reranker
- 记录所有指标
- 特别关注：reranker增加了多少latency？值不值得？

**关键产出：** Reranker对比数据 + latency tradeoff初步数据

### Day 5: Chunking策略对比

**上午：实现3种chunking**
- Fixed-size chunking (512 tokens, 256 overlap)
- Recursive character split (LangChain style)
- Sentence-based chunking (按句子边界切分)
- 对每种chunking重建FAISS索引

**下午：ablation实验**
- 3种chunking × 最优retrieval策略
- 记录Recall@5变化
- 分析：什么样的chunk大小在什么数据集上效果最好？

**关键产出：** Chunking ablation表格

---

## Phase 3: Query Expansion（Day 6-7）

### Day 6: HyDE + Query Rewriting

**上午：实现HyDE**
- 用LLM生成hypothetical answer
- 对hypothetical answer做embedding
- 用这个embedding去做retrieval
- 实现为query transformation模块

**下午：实现Query Rewriting**
- 用LLM将用户query改写为更精确的搜索query
- 对multi-hop问题：尝试query decomposition（将复合问题拆成子问题）
- 注意：这里只做简单的single-step decomposition，不做iterative

**关键产出：** 两种query expansion实现

### Day 7: Query Expansion实验

**上午：跑实验**
- 在最优retrieval配置(Hybrid + Reranker)上叠加：
  - + HyDE
  - + Query Rewriting
  - + Query Decomposition (仅HotpotQA)
- 记录所有指标

**下午：分析query expansion的tradeoff**
- Accuracy提升了多少？
- 额外的LLM调用增加了多少latency和cost？
- 哪类query受益最大？（这个发现很重要）
- 初步整理到目前为止的所有实验数据

**关键产出：** Query expansion对比数据 + tradeoff分析

---

## Phase 4: Evaluation框架 + 全量实验（Day 8-9）

### Day 8: 完善评估体系

**上午：实现额外指标**
- Answer Faithfulness：用LLM-as-judge检查answer是否被retrieved context支持
  - Prompt：给定context和answer，判断answer中每个claim是否有context支持
  - 输出：faithfulness score (0-1)
- Hallucination Rate：faithfulness < 0.5的比例
- 接入RAGAS框架做交叉验证（如果时间够的话）

**下午：全量实验矩阵**
- 跑完所有组合的最终实验：

| Config | Retriever | Reranker | Query Expansion | Chunking |
|--------|-----------|----------|-----------------|----------|
| C1 | Dense | ✗ | ✗ | 512 fixed |
| C2 | Hybrid | ✗ | ✗ | 512 fixed |
| C3 | Hybrid | ✓ | ✗ | 512 fixed |
| C4 | Hybrid | ✓ | HyDE | 512 fixed |
| C5 | Hybrid | ✓ | Rewrite | 512 fixed |
| C6 | Hybrid | ✓ | ✗ | Recursive |
| C7 | Hybrid | ✓ | ✗ | Sentence |

- 确保所有中间结果都保存

**关键产出：** 完整实验矩阵 + faithfulness指标

### Day 9: Latency + Cost分析

**上午：Latency profiling**
- 对每种配置，分解latency到各环节：
  - Retrieval time
  - Reranking time
  - Query expansion time (LLM call)
  - Generation time (LLM call)
- 画 accuracy vs latency scatter plot

**下午：Cost分析**
- 统计每种配置的：
  - LLM API调用次数 per query
  - 平均input/output tokens per query
  - 估算cost per 1000 queries
- 画 performance gain per dollar 图表
- 生成"什么场景下用什么策略"的推荐矩阵

**关键产出：** Latency分解表 + Cost分析 + 推荐矩阵

---

## Phase 5: Error Analysis + Failure Mode（Day 10-11）

### Day 10: Failure Mode分类

**上午：实现自动化failure分类**
- 对每个错误答案，自动判断failure type：
  1. Coverage failure：gold document不在corpus中（检查gold doc ID）
  2. Recall failure：gold doc在corpus中但没被检索到（top-20中没有）
  3. Ranking failure：gold doc被检索到但排名太低（在top-20但不在top-5）
  4. Generation failure：正确doc被检索到但LLM答错了

**下午：生成failure breakdown表**
- 对每种retrieval策略，统计4类failure的比例
- 生成GPT建议的那张表：

| Failure Type | Naive | Hybrid | Hybrid+Rerank | +HyDE |
|---|---|---|---|---|
| Coverage | ?% | ?% | ?% | ?% |
| Recall | ?% | ?% | ?% | ?% |
| Ranking | ?% | ?% | ?% | ?% |
| Generation | ?% | ?% | ?% | ?% |

- 分析：每种改进主要解决了哪类failure？

**关键产出：** Failure mode breakdown表 + 分析

### Day 11: Qualitative Analysis + 洞察总结

**上午：选代表性case做深度分析**
- 从每类failure中选2-3个典型case
- 分析具体的query、retrieved docs、和generated answer
- 特别关注：
  - Hybrid search在哪类query上比pure dense好？（entity name? rare term?）
  - Reranker对什么类型的doc排序改善最大？
  - HyDE在哪里帮倒忙了？（hallucination风险）

**下午：汇总所有发现**
- 写一份findings summary：
  - 核心发现1：Hybrid search对containing rare entities的query提升最大（+X%）
  - 核心发现2：Reranker对multi-hop问题贡献显著（+X%），但增加Y ms latency
  - 核心发现3：HyDE提升了Z%的recall但增加了W%的hallucination risk
  - 核心发现4：在cost-constrained场景下，Hybrid+Reranker是最优性价比选择
- 这份summary就是你面试时讲的核心内容

**关键产出：** Case study + Findings summary

---

## Phase 6: 前端 + 文档 + 包装（Day 12-13）

### Day 12: Streamlit前端 + 可视化

**上午：Streamlit交互界面**
- 左侧：选择retrieval策略、chunking方式、是否开启reranker/expansion
- 中间：输入query，显示answer + 引用的source chunks
- 右侧：显示retrieved chunks的relevance score
- 底部：显示latency breakdown

**下午：结果可视化Dashboard**
- 实验结果对比图表（用Plotly或Matplotlib）：
  - Accuracy对比柱状图
  - Accuracy vs Latency scatter
  - Failure mode stacked bar chart
  - Cost comparison table
- 生成所有图表的静态版本用于README

**关键产出：** 可交互的demo + 可视化图表

### Day 13: GitHub包装

**上午：README撰写**
- Architecture diagram（用Mermaid或draw.io）
- 安装步骤
- Quick start
- 实验结果表格 + 图表
- Key findings总结

**下午：代码清理 + Docker**
- 代码review和refactor
- 添加type hints和docstrings
- 写Dockerfile + docker-compose
- 确保 `docker-compose up` 能一键启动

**关键产出：** 完整的GitHub项目 + Docker化

---

## Optional: Day 14

如果有余力，选一个做：

**选项A：简化版Graph-enhanced（推荐度：中）**
- 用LLM从query中抽取entities
- 在retrieved chunks中找包含这些entities的额外chunks
- 本质上是entity-aware retrieval，不是完整GraphRAG
- 作为实验矩阵中的一行数据

**选项B：技术博客（推荐度：高）**
- 把findings summary扩展成一篇Medium/个人博客文章
- 标题类似："I Benchmarked 7 RAG Configurations on 3 Datasets — Here's What Actually Matters"
- 配上你的可视化图表
- 这篇文章本身就是面试谈资

**选项C：Embedding模型对比（推荐度：中）**
- 对比SBERT vs OpenAI text-embedding-3-small vs BGE
- 作为ablation的额外维度

---

## 技术栈总结

| 组件 | 选型 |
|------|------|
| Embedding | SBERT (all-MiniLM-L6-v2)，可选OpenAI |
| Sparse Retrieval | rank_bm25 或 Pyserini |
| Dense Retrieval | FAISS (IVF索引) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Claude API / OpenAI API |
| Evaluation | 自实现 + RAGAS |
| 数据集 | FlashRAG预处理版 / HuggingFace |
| 前端 | Streamlit |
| 可视化 | Plotly / Matplotlib |
| 部署 | Docker + docker-compose |

---

## 面试时的讲述框架（3分钟版）

**开头（30秒）：** "我做了一个模块化的RAG系统，在3个标准benchmark上系统性地评估了不同retrieval策略的效果，并从accuracy、latency、cost三个维度做了工程权衡分析。"

**技术细节（1分钟）：** "我对比了7种配置组合，包括dense vs hybrid search、有无cross-encoder reranking、不同query expansion策略。核心发现是hybrid search + reranker是性价比最高的配置，在HotpotQA上比naive baseline提升了X% Recall@5，而latency只增加了Y ms。"

**深度展示（1分钟）：** "我还做了failure mode分析，把错误分成coverage、recall、ranking、generation四类。发现hybrid search主要解决的是recall failure，从X%降到Y%，但generation failure在所有配置中都保持在Z%左右，说明retrieval质量到达某个阈值后，bottleneck转移到了generation端。"

**收尾（30秒）：** "这个项目让我深度理解了RAG系统的每个组件如何影响最终效果，以及在实际部署时如何根据latency和cost约束选择最优配置。"
