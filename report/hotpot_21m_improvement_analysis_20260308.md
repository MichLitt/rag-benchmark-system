# HotpotQA 21M 检索优化改进分析报告

## 1. 背景与开始状态

本项目最初的目标，是在本地 RAG Benchmark 框架中搭建一个可复现、可扩展的检索与生成实验平台，并重点观察不同检索策略在 `HotpotQA`、`NQ`、`TriviaQA` 等数据集上的表现。

在进入本轮改进之前，项目已经具备以下基础能力：

- 已有 `Naive RAG` 主流程，支持 `dense / bm25 / hybrid` 检索。
- 已有 `reranker`、`dedup`、`query expansion`、延迟统计、token/cost 统计。
- 已有 `HotpotQA` 专项检索评估脚本与部分失败分析工具。
- 已能运行 `extractive` 和 `openai_compatible` 两类生成后端。

但在实际实验中，`HotpotQA` 的 `F1` 和 `Recall` 长期偏低，主要暴露出两个问题：

1. **语料覆盖不足**
   - 早期 `1M` 级别语料对 `HotpotQA` 的 gold pages 覆盖严重不足。
   - 在 `1M` 阶段的 `50` 条检索评估中，指标只有：
     - `CoverageAny = 0.24`
     - `CoverageAll = 0.04`
     - `RecallAnyGoldTitle@k = 0.14`
     - `RecallAllGold@k_title = 0.04`
   - 这说明大量问题连 supporting pages 都不在索引里，后续 rerank 或 LLM 基本无从发挥。

2. **多跳检索能力不足**
   - 即使 gold titles 部分存在，系统也更容易只捞到一个 supporting page，而不是同时捞到两篇。
   - `HotpotQA` 的核心难点不是“命中任意相关文档”，而是“同时命中两个 supporting titles”。

因此，本轮改进的核心判断是：

- 第一阶段先解决语料覆盖，把 `HotpotQA` 从“缺料”状态拉到“可检索”状态。
- 第二阶段再把优化目标从 `RecallAnyGoldTitle@k` 切到 `RecallAllGold@k_title`，专门处理多跳检索。

---

## 2. 本轮主要修改

### 2.1 构建 21M 分片 Dense 索引

为避免 21M 级别索引构建导致内存或磁盘失控，项目新增了安全的分片构建方案：

- 新增 `scripts/build_dense_sharded_index.py`
- 新增 `dense_sharded` 检索模式
- 新增 root manifest 与分片 manifest 结构
- 每个 shard 独立保存：
  - `faiss.index`
  - `docstore.jsonl`
  - `docstore_offsets.bin`
  - `dense_config.json`
- docstore 改为 offsets 懒加载，不再在初始化时全量载入内存

最终构建出的正式索引规模为：

- `total_docs = 21,015,324`
- `shard_size = 1,000,000`
- 总计 `22` 个 shard
- embedding model 为 `sentence-transformers/all-MiniLM-L6-v2`

这一步的意义是把系统从“语料覆盖严重不足”推进到“语料基本够用”。

### 2.2 建立 21M 下的统一 retrieval-only 评估链路

为了把“语料覆盖问题”和“检索/排序问题”拆开，项目补充并固化了以下评估能力：

- `scripts/eval_hotpot_retrieval.py`
  - 支持 `dense_sharded`
  - 输出 `CoverageAny / CoverageAll / RecallAnyGoldTitle@k / RecallAllGold@k_title`
- `scripts/build_hotpot_coverage_filtered_subset.py`
  - 生成 all-gold-covered 子集
  - 用于测量“在语料完全覆盖前提下，检索本身能做到什么程度”

这使得后续分析可以明确区分：

- 语料覆盖上限
- 检索召回能力
- rerank 对双页命中的影响

### 2.3 把 Hotpot 检索主线切到 title-first

在 21M 覆盖基本补齐后，本轮继续实现了针对 `HotpotQA` 的 title-first 候选整形策略：

- 默认配置切到：
  - `top_k = 20`
  - `retrieve_top_k = 150`
  - `dedup_mode = title`
  - `dedup_before_rerank = true`
  - `title_first_rerank = true`
- 新增 title-first 处理逻辑：
  - raw retrieval 深召回
  - 按 `normalized title` 分组
  - 每个 title 只保留最高分代表 chunk 进入 reranker
  - rerank 后再按 title-diverse 规则回填 chunk
- 新增可配置参数：
  - `title_pool_k = 40`
  - `max_chunks_per_title = 2`
  - `min_unique_titles = 6`

这一步的目标，不是让同一页面的多个 chunk 霸占候选池，而是尽量把第二篇 supporting page 保住。

### 2.4 强化 Hotpot 双 gold 诊断与失败分类

为了判断双页命中到底丢在哪个阶段，项目补充了更细粒度的诊断字段：

- `gold_title_ranks`
- `gold_titles_in_raw_candidates`
- `gold_titles_after_dedup`
- `gold_titles_in_final_top_k`
- `missing_gold_count`
- `first_gold_found`
- `second_gold_found`
- `retrieval_failure_bucket`

失败分类也改成了更符合 `HotpotQA` 的结构：

- `no_gold_in_raw`
- `only_one_gold_in_raw`
- `both_gold_in_raw_but_lost_after_dedup`
- `both_gold_after_dedup_but_lost_after_rerank`
- `both_gold_in_final`
- `generation_failure`

这使得分析不再停留在“Recall 低”，而是能定位到问题是：

- 原始召回就不够
- 去重时丢掉了 gold
- rerank 把双页结构打散了
- 检索没问题，但生成失败

### 2.5 配置与实验入口整理

本轮同时整理了正式实验入口：

- 更新 `config/wiki18_21m_sharded.yaml`
  - 作为当前 `HotpotQA retrieval-only` 主配置
- 新增 `config/wiki18_21m_sharded_llm.yaml`
  - 为后续真实 `openai_compatible` 生成实验预留
- `run_naive_rag_baseline.py`
  - 接入新检索参数
  - 落盘新诊断字段
- `analyze_failure_modes.py`
  - 接入新的 retrieval failure buckets

---

## 3. 最终结果

### 3.1 21M 检索正式结果

在 `21M dense_sharded` 正式索引上，对 `HotpotQA` 全量前 `200` 条样本进行 retrieval-only 评估，结果如下：

| 指标 | 1M 阶段基线（50 query） | 21M 正式索引（200 query） |
| --- | ---: | ---: |
| CoverageAny | 0.24 | 0.94 |
| CoverageAll | 0.04 | 0.645 |
| RecallAnyGoldTitle@k | 0.14 | 0.65 |
| RecallAllGold@k_title | 0.04 | 0.185 |
| AvgRetrievalLatencyMs | 72.38 ms | 1321.67 ms |

结论很明确：

- `21M` 把 `CoverageAny` 从 `0.24` 拉到了 `0.94`
- `CoverageAll` 从 `0.04` 拉到了 `0.645`
- `RecallAllGold@k_title` 从 `0.04` 提升到 `0.185`

也就是说，语料不足这个主问题已经被大幅缓解。

### 3.2 coverage-filtered 子集结果

在 `21M` 语料上，构造 all-gold-covered 子集后得到：

- 总样本数：`7405`
- 保留样本数：`4914`
- 保留比例：`66.36%`

在这个 coverage-filtered 子集上的 retrieval-only 结果为：

| 指标 | coverage-filtered 结果 |
| --- | ---: |
| CoverageAny | 1.0 |
| CoverageAll | 0.9994 |
| RecallAnyGoldTitle@k | 0.8488 |
| RecallAllGold@k_title | 0.3128 |
| AvgRetrievalLatencyMs | 1591.70 ms |

这个结果说明：

- 当 gold pages 基本都在语料里时，系统命中任意一个 supporting page 已经比较强
- 但同时命中两个 supporting pages 的能力仍然明显不够

### 3.3 最新检索策略与诊断能力

在本轮 title-first 改造后，项目已经具备以下新默认策略：

- `retrieve_top_k = 150`
- `dedup_mode = title`
- `dedup_before_rerank = true`
- `title_first_rerank = true`
- `title_pool_k = 40`
- `max_chunks_per_title = 2`
- `min_unique_titles = 6`

此外，新的诊断与失败分类链路已经可用，并已完成最小样本集成验证：

- `run_naive_rag_baseline.py` 可输出新字段
- `eval_hotpot_retrieval.py` 可输出增强版 `details.json`
- `analyze_failure_modes.py` 可按新 bucket 分类
- 单元测试 `27` 项全部通过

---

## 4. 对结果的分析

### 4.1 已经解决的问题

本轮最重要的成果，是把项目从“覆盖不足导致实验无意义”推进到了“覆盖基本够用，可以认真优化检索”的阶段。

之前即使继续调 reranker、prompt 或 query expansion，也很难有稳定收益，因为 supporting pages 大量不在语料中。现在这个问题已经不是主矛盾了。

### 4.2 当前剩余的主瓶颈

目前最核心的瓶颈是：

**第二篇 supporting page 进不来，或者进来之后被排序挤掉。**

证据主要有两点：

1. 在全量 `200` 条样本上：
   - `CoverageAll = 0.645`
   - 但 `RecallAllGold@k_title = 0.185`

2. 在 coverage-filtered 子集上：
   - `CoverageAll ≈ 1.0`
   - 但 `RecallAllGold@k_title = 0.3128`

这说明：

- 不是语料里没有
- 而是 retrieval/rerank 没有稳定把两篇都保留下来

### 4.3 对 F1 的判断

当前阶段不应过早拿 `F1` 作为主判断指标，原因有二：

1. 当前主实验仍然以 retrieval-only 为主，很多 run 不是完整 LLM 生成；
2. `extractive` baseline 会把第一段 passage 当作答案代理，这天然会拉低 token-level `F1`。

因此，当前更合理的路径是：

- 先把检索指标打到位，尤其是 `RecallAllGold@k_title`
- 再进入真实 `openai_compatible` 的小规模端到端实验

### 4.4 对 title-first 改造的评价

从方法上看，title-first 是正确方向，原因是它直接针对 `HotpotQA` 的结构性问题：

- 同一 title 的重复 chunk 会挤占候选预算
- 多跳题真正需要的是“保住 title 多样性”
- reranker 如果直接吃原始 chunk 列表，很容易把排序进一步压成单页高密度

所以，title-first 的意义不是简单“换一个排序方法”，而是把优化目标从“相似 chunk 排得更靠前”改成“两个 supporting titles 尽量都留在系统里”。

### 4.5 下一步建议

基于当前结果，下一阶段最合理的工作应是：

1. 在 `21M` 上跑完整 A/B 检索实验矩阵  
   固定 `dense_sharded`，比较：
   - `retrieve_top_k = 50 / 100 / 150`
   - `title_first_rerank = off / on`
   - `retriever_rank_weight = 0.0 / 0.2 / 0.4`

2. 用增强后的 `retrieval_failure_bucket` 做误差归因  
   重点看：
   - 双 gold 是否已经进入 raw candidates
   - 是否在 dedup 阶段丢失
   - 是否在 rerank 阶段丢失

3. 只有在 raw retrieval 双 gold 仍明显不足时，再引入 `hotpot_decompose`  
   当前不应优先使用 `hyde`

4. 当 `RecallAllGold@k_title` 达到更高水平后，再进入真实 LLM 生成实验  
   否则端到端 `F1` 仍然会被检索瓶颈主导

---

## 5. 总结

本轮改进的总体结论是：

- **语料覆盖问题已基本被解决**
- **项目已从“缺料导致结论不可信”进入“可以认真优化多跳检索”的阶段**
- **当前真正的主瓶颈是双 supporting pages 的联合召回与排序保留**

如果用一句话概括当前状态：

> 项目已经从“能跑实验”推进到了“可以做有意义的 Hotpot 多跳检索优化”，但还没有到“可以直接用端到端 F1 评价系统优劣”的阶段。
