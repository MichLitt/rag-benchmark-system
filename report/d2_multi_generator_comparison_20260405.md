# D2 多生成器对比实验：技术分析报告

**实验日期**：2026-04-05
**报告版本**：v1.0
**实验阶段**：Phase 2 / D2（探索性目标）

---

## 1. 实验设计

### 1.1 目标

分离检索贡献与生成贡献，量化不同大模型在固定检索管道下对 RAG 端到端指标的影响。

### 1.2 控制变量（固定）

所有配置共享同一检索管道：

| 参数 | 值 |
|------|-----|
| 检索器 | `dense_sharded`（Wiki18 21M，分片 FAISS） |
| 重排序器 | Cross-Encoder `ms-marco-MiniLM-L-6-v2` |
| top\_k（最终上下文） | 5 |
| retrieve\_top\_k（候选池） | 100 |
| dedup\_mode | `title` |
| Query Expansion | off |
| 每数据集样本量 | 200 |

### 1.3 实验变量（生成器）

| 实验组 | 模型 | SDK / 协议 | max\_output\_tokens |
|--------|------|-----------|---------------------|
| Extractive（对照） | 无 LLM，取检索文本首段 | — | — |
| MiniMax-M2.5（Phase 4 基准） | MiniMax-M2.5 | OpenAI-compatible | 128 |
| MiniMax-M2.7 | MiniMax-M2.7 | **Anthropic SDK**（extended thinking 模型） | 4096 |
| GLM-5.1 | GLM-5.1 | OpenAI-compatible（api.z.ai coding plan） | 256（text）/ 2048（reasoning） |

### 1.4 数据集

| 数据集 | 类型 | 评测指标 |
|--------|------|---------|
| HotpotQA | 多跳推理 QA | EM, F1, title Recall@5 |
| NQ（Natural Questions） | 单跳事实 QA | EM, F1, answer-presence Recall@5 |
| TriviaQA | 百科知识 QA | EM, F1, answer-presence Recall@5 |

---

## 2. 实验结果

### 2.1 主指标汇总（EM / F1）

| 模型 | HotpotQA EM | HotpotQA F1 | NQ EM | NQ F1 | TriviaQA EM | TriviaQA F1 | **宏平均 EM** | **宏平均 F1** |
|------|------------|------------|-------|-------|------------|------------|------------|------------|
| Extractive | 0.000 | 0.019 | 0.000 | 0.026 | 0.000 | 0.037 | **0.000** | **0.027** |
| MiniMax-M2.5（P4 基准） | 0.045 | 0.138 | 0.010 | 0.113 | 0.045 | 0.187 | **0.033** | **0.146** |
| MiniMax-M2.7 | 0.080 | 0.171 | 0.010 | 0.126 | 0.110 | 0.249 | **0.067** | **0.182** |
| **GLM-5.1** | **0.190** | **0.293** | **0.165** | **0.327** | **0.400** | **0.504** | **0.252** | **0.375** |

### 2.2 检索指标（所有模型一致）

| 数据集 | Recall@5 |
|--------|---------|
| HotpotQA | 0.655 |
| NQ | 0.640 |
| TriviaQA | 0.615 |

> 检索 Recall@5 完全相同，证明 §2.1 中的 EM/F1 差距 **纯由生成器决定**。

### 2.3 运行时指标

| 模型 | 数据集 | 平均生成延迟 | 错误率 | 错误类型 |
|------|--------|------------|--------|---------|
| Extractive | 全部 | 0 ms | 0% | — |
| MiniMax-M2.7 | HotpotQA | 16,096 ms | 2.0% | ThinkingBlock 预算不足 |
| MiniMax-M2.7 | NQ | 7,625 ms | 0.0% | — |
| MiniMax-M2.7 | TriviaQA | 10,804 ms | 1.0% | ThinkingBlock 预算不足 |
| GLM-5.1 | HotpotQA | 27,170 ms | **20.5%** | 超时(13) + budget 耗尽(28) |
| GLM-5.1 | NQ | 19,622 ms | **8.5%** | 超时(10) + budget 耗尽(6) |
| GLM-5.1 | TriviaQA | 21,756 ms | **13.0%** | 超时(8) + budget 耗尽(15) |

> **注**：GLM-5.1 的 EM/F1 是在有效生成的样本（含 `generated_answer`）上统计的，若将失败样本计为 0，则实际分数更低。

---

## 3. 深度分析

### 3.1 生成能力才是 RAG 瓶颈，不是检索

所有模型共享相同的检索管道（Recall@5 固定），但 EM 跨度从 0.000 到 0.400，F1 跨度从 0.019 到 0.504。**生成器质量是端到端性能的决定性因素**。

Phase 1 的关键结论认为"生成是真正瓶颈"（Hallucination Rate 88-97%，F1 最高 0.21），本次实验在完全不改变检索的前提下将 F1 提升至 0.504（GLM-5.1 / TriviaQA），**直接实证验证了该结论**。

### 3.2 GLM-5.1 显著领先，TriviaQA 表现惊艳

GLM-5.1 在三个数据集上均大幅领先：

- **TriviaQA**：EM=0.400，F1=0.504。达到了本项目迄今为止的最高水平。
- **HotpotQA**：EM=0.190，是多跳推理场景下的突破（Phase 4 最优 0.055）。
- **NQ**：EM=0.165，相比 MiniMax-M2.5（0.010）提升 16.5x。

GLM-5.1 在 TriviaQA 上的优势尤为突出，推测原因：
1. TriviaQA 以**事实性百科知识**为主，与 GLM 系列训练数据重合度高。
2. GLM-5.1 支持推理链（reasoning），对提取事实有帮助。
3. TriviaQA 答案通常较短（实体、日期、人名），与模型输出格式契合。

### 3.3 MiniMax-M2.7 相对 M2.5 有适度提升

MiniMax-M2.7（Anthropic extended thinking 模型）在所有数据集上均优于 M2.5：

| 数据集 | M2.5 F1 | M2.7 F1 | 提升 |
|--------|---------|---------|------|
| HotpotQA | 0.138 | 0.171 | +24% |
| NQ | 0.113 | 0.126 | +12% |
| TriviaQA | 0.187 | 0.249 | +33% |

提升幅度较为温和，可能原因：
- Extended thinking 模型消耗大量 token 做推理，在短答案场景（RAG）中边际收益有限。
- 当前 `max_output_tokens=4096` 保证了 ThinkingBlock 有足够空间，但生成延迟显著（16s vs 5.8s for 旧配置的超时失败版本）。

### 3.4 GLM-5.1 的错误率是关键隐患

GLM-5.1 的生成错误率在 HotpotQA 高达 **20.5%**（41/200），由两类原因构成：

| 错误类型 | HotpotQA | NQ | TriviaQA | 原因 |
|---------|----------|----|---------|----|
| TimeoutError | 13 | 10 | 8 | 60s 超时不足，GLM 复杂推理耗时更长 |
| Budget Exhausted | 28 | 6 | 15 | `max_completion_tokens=2048` 对深度推理不够 |

这意味着 GLM-5.1 的实测 EM/F1 是**有利偏差**的：仅在成功生成的样本上统计，真实全集分数更低。若将失败样本计为空答案（EM=0, F1=0），调整后指标为：

| 数据集 | GLM-5.1 报告 EM | 调整后 EM | GLM-5.1 报告 F1 | 调整后 F1 |
|--------|--------------|---------|--------------|---------|
| HotpotQA | 0.190 | ~0.151 | 0.293 | ~0.233 |
| NQ | 0.165 | ~0.151 | 0.327 | ~0.299 |
| TriviaQA | 0.400 | ~0.348 | 0.504 | ~0.439 |

即便调整后，GLM-5.1 仍大幅领先其他模型。

### 3.5 Extractive 方法的局限性

Extractive（直接取检索结果首段）在所有数据集上 EM=0，F1 < 0.04。根本原因：
1. 检索到的 chunk 通常是段落级，而 gold answer 是短答案（几个词/一句话）。
2. F1 计算是词级别 overlap，段落与短答案的 overlap 本就极低。
3. 无法做跨文档推理（HotpotQA 多跳）。

Extractive 的 F1（0.019–0.037）作为**生成下界**，说明生成器引入了实质性的理解与提炼能力。

---

## 4. 模型选型建议

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 事实类 QA（TriviaQA 类型） | GLM-5.1 | EM/F1 最高，知识覆盖广 |
| 多跳推理（HotpotQA 类型） | GLM-5.1 | EM=0.190，远超其他模型 |
| 低延迟场景 | MiniMax-M2.7 | 平均生成延迟 8–16s，低于 GLM（20–27s） |
| 零错误率需求 | MiniMax-M2.7 | 错误率 0–2%；GLM 需要提高 timeout 和 budget |

### 后续配置优化建议（GLM-5.1）

```yaml
generation:
  timeout_sec: 180          # 从 60s 提升到 180s
  max_completion_tokens: 8192  # 从 2048 提升到 8192
```

预计可将错误率从 8.5–20.5% 降至 < 3%，届时 GLM-5.1 的真实有效指标将进一步提升。

---

## 5. 实验工程记录

### 5.1 发现并修复的 Bug

**Bug 1：`max_output_tokens=256` 对 MiniMax-M2.7 不够**
MiniMax-M2.7 使用 extended thinking，优先消耗 ThinkingBlock，256 token 全部用尽后没有剩余空间生成 TextBlock，导致 `RuntimeError: Anthropic response did not contain a text content block.`。

- 旧运行（174326）：hotpotqa 错误率 45%，nq 27%，triviaqa 30.5%，**所有答案均为空**。
- 修复：`max_output_tokens: 256 → 4096`，错误率降至 0–2%。
- **根因修复已合入 `AnthropicCompatibleGenerator`**：跳过 ThinkingBlock，取第一个 `type=="text"` 的块。

**Bug 2：`run_naive_rag_baseline.py` 不加载 `.env`**
脚本启动时不调用 `load_dotenv()`，导致 `ANTHROPIC_API_KEY` 读不到，MiniMax 第一次运行直接崩溃。

- 修复：在 `run_naive_rag_baseline.py` 和 `main.py` 头部加 `load_dotenv(override=True)`。

### 5.2 实验运行目录映射

| 运行目录 | 模型 | 状态 |
|---------|------|------|
| `161143_rerank` | Extractive | ✅ 有效 |
| `161944_rerank` | GLM-5.1 | ✅ 有效（含高错误率） |
| `174326_rerank` | MiniMax-M2.7 (旧，256 token) | ❌ 作废 |
| `191610_rerank` | MiniMax-M2.7 (修复，4096 token) | ✅ 有效 |

---

## 6. 结论

1. **GLM-5.1 是当前最强生成器**，三数据集平均 EM=0.252、F1=0.375，TriviaQA 达到 EM=0.400，是本项目最高记录。
2. **生成器质量是 RAG 系统端到端性能的主导因素**——相同检索、不同生成器，F1 相差 14 倍。
3. **MiniMax-M2.7 适合低延迟、高稳定性场景**，性能中等但错误率极低。
4. **GLM-5.1 需要调整超时和 budget 参数**才能发挥完整潜力；当前 20.5% 错误率对生产场景不可接受。
5. Extractive 方法在固定 top_k=5 场景下几乎无效，说明 RAG 中的 LLM 生成环节不可或缺。

---

*生成时间：2026-04-05*
*实验代码：`scripts/run_naive_rag_baseline.py`*
*配置文件：`config/d2_extractive.yaml`, `config/d2_glm51.yaml`, `config/d2_minimax_m27.yaml`*
