# Phase 2 计划：Agent Knowledge and Document Intelligence System

**文档版本**：v1.3（2026-04-04，自检统一性与完备性修订）
**撰写日期**：2026-04-04
**项目状态**：Phase 1 已关闭（2026-03-10），本文件规划 Phase 2 转型路径
**对应路线图**：`AI_Intern_Project_Roadmap.md` §6 / Phase 3（第 6-7 周）

> **v1.1 变更说明**：经第一轮 Codex 审查，修正了以下问题：Document schema 迁移缺失、ingest API 与现有 index builder 不兼容、Vectara 模型加载代码错误、SPLADE++ 年份错误、FaithBench 引用范围偏差、SPLATE 延迟声明过度推广、citation 指标缺乏操作性定义、page grounding 精确匹配过严、人工一致性样本量不足、B1/B2 scope 无 gate、两周时序切分不合理。

> **v1.2 变更说明**：经第二轮 Codex 深度代码审查（读取真实 src/ 文件），修正了以下问题：A0 迁移范围补全（加入 build_retrieval_indexes.py / build_dense_sharded_index.py / corpus.py）；citation 指标重定义为 post-hoc NLI attribution（非 LLM 输出提取）；ScoredDocument 改为 adapter 模式避免跨切重构；LazyDocstore 并发 bug 修复方案明确；/ingest index_id 映射与 manifest 规范补全；EvalOps 上报移入 orchestration 层；下游 Phase 1 分析脚本兼容性纳入范围；page_number 单字段改为 page_start/page_end；2 周范围进一步收窄为 read-only API + 离线 NLI 实验；面试叙事 NLI 措辞修正。

> **v1.3 变更说明**：自检统一性与完备性，修正了以下内部矛盾：§2.1 MVP 目标清单与正文对齐（page_start/page_end、ScoredDocument 层次、/ingest stretch 标注、指标名称）；A1 流程图和验收标准字段名同步；B1/D3 代码示例改为返回 List[Document]（符合 adapter 原则）；EvalRunReport 字段名与 §3.1/A3 指标名统一；§5 目录结构图补充 corpus.py、修正 cross_encoder.py 状态为"修改"、修正 pipeline.py 注释；§9 release gate 指标名更新；§10 面试叙事 /ingest 标注为 stretch；C1/C2 之间补充分隔线。

---

## 1. 背景与定位转变

### 1.1 Phase 1 完成情况回顾

Phase 1 的核心 RAG benchmark 工作已于 2026-03-10 完整关闭，主要结论如下：

| 维度 | 最终结论 |
|------|----------|
| 检索天花板 | `RecallAllGold@k_title = 0.235`（flat 21M dense_sharded，200q） |
| 主要检索 blocker | `query_formulation_gap`（64/200）、`budget_limited`（45/200） |
| IVF 加速 | 速度提升 100x，但 recall gate 未通过，已 reject |
| 生成质量 | F1 最高 0.21，Hallucination Rate 88-97%，生成是真正瓶颈 |
| 忠实度评分可信度 | 单一 LLM judge 信度有限（RAGTruth 2024 研究表明 judge 间分歧显著） |
| 已关闭的支线 | IVF4096 / title_prefilter / 无限 recall ablation |

> **注**：原 v1.0 中引用 FaithBench（arXiv:2505.04847）支持"单一 LLM judge ~50% 准确率"。FaithBench 主要面向摘要幻觉检测场景，不直接等同于 RAG citation grounding。修订后改用 RAGTruth（IEEE S&P 2024）和 HHEM（Vectara）作为更贴合 RAG 场景的引用依据。

**关键判断**：继续在 HotpotQA recall 上做 ablation 的边际收益极低。下一步应切换赛道。

### 1.2 战略转型：从 RAG Benchmark 到 Agent Knowledge Subsystem

根据 `AI_Intern_Project_Roadmap.md` 的整体策略调整：

```
旧定位：独立的 RAG benchmark flagship
新定位：为 agent 提供复杂知识上下文处理能力的文档理解与检索子系统
```

具体含义：

- **不再**以提升 HotpotQA recall 为主线目标
- **转型为**可被 `llm-coding-agent-system` 调用的知识服务
- **新增**真实文档（PDF）处理能力，这是纯文本 benchmark 不具备的
- **增加** citation grounding 指标，与现有 EM/F1 **并列**（非取代）
- **接入** `llm-evalops-platform`，让结果可追踪、可 release gate

### 1.3 时序位置与周内切分

```
Week 1-3   Agent Runtime (llm-coding-agent-system 重构)
Week 4-5   EvalOps Platform (新 repo 搭建)
Week 6-7   ★ 本计划：rag-benchmark-system 转型           ← 当前阶段
Week 8     coding-llm-finetune 收敛
Week 9     统一包装 / 简历 bullet / 面试准备
```

**两周内部切分（v1.2 收窄版）**：

```
Week 6（Days 1-7）：schema 迁移 + ingestion
  Day 1-2: A0 Document schema 迁移（src/types.py + docstore.py +
            corpus.py + build_retrieval_indexes.py + build_dense_sharded_index.py）
           + test_docstore_migration.py 通过
  Day 3-4: A1 PDF ingestion CLI（pdfplumber，native-text only）
           + TokenAwareChunker（page_start/page_end）
  Day 5-6: A3 离线 NLI attribution（HHEMScorer + CitationEvaluator）
           对 toy PDF Q&A 验证 HHEM 加载与 Cohen's Kappa
  Day 7:   buffer / 测试修复

Week 7（Days 8-14）：API + EvalOps + Phase 1 兼容
  Day 8-9: A2 read-only /retrieve API（基于已有 index，修复 LazyDocstore 并发）
  Day 10:  A2 /ingest job（若时间允许），否则仅 /retrieve
  Day 11:  C1/C2 EvalOps 上报（orchestration 层，兼容旧 JSON 格式）
  Day 12:  C2 下游脚本兼容性更新（aggregate / export / score_faithfulness）
  Day 13-14: B1/B2 Gate 检查 → 时间充裕则启动 SPLADE 或 setwise
             否则直接进入 Phase 5 实验矩阵

可达成的最小 2 周交付物：
  ✓ A0 schema 迁移 + 向后兼容
  ✓ A1 native-text PDF ingestion CLI
  ✓ A3 离线 NLI attribution（HHEM 验证通过）
  ✓ A2 read-only /retrieve API（并发安全）
  ✓ C1 EvalOps orchestration 层上报

超出 2 周的内容（列为 stretch）：
  △ /ingest async job + manifest 系统
  △ labeled PDF dataset with 100+ Q&A pairs
  △ B1 SPLADE / B2 setwise reranking
```

---

## 2. Phase 2 目标

### 2.1 核心 MVP 目标（必须完成）

1. **Document schema 迁移（A0，强前置）**：扩展 `src/types.py` 的 `Document` 类，新增 `page_start / page_end / section / source` 字段；同步更新 `docstore.py`、`corpus.py`、`build_retrieval_indexes.py`、`build_dense_sharded_index.py`，覆盖全部序列化路径，保持向后兼容
2. **文档 ingestion pipeline（native-text PDF）**：使用 `pdfplumber` 解析非扫描 PDF，产出带 `page_start/page_end` 的 chunk；OCR 支持为 stretch goal
3. **ScoredDocument（API 层 adapter）**：在 `src/api/` 层引入 `ScoredDocument` 包装类型，不修改现有 `pipeline.py` / `cross_encoder.py` / `hybrid.py` 的内部接口
4. **检索 API 化（read-only /retrieve 为 MVP）**：封装为 FastAPI 服务，Week 7 必须交付 `/retrieve`；`/ingest` 异步 job 为 stretch，时间充裕时实现
5. **NLI post-hoc attribution 评测**：新增 `answer_attribution_rate`、`supporting_passage_hit`、`page_grounding_accuracy` 三项指标，与 EM/F1 **并列使用**（非取代）；仅适用于 PDF Q&A 数据集
6. **NLI-based 忠实度评分**：验证 Vectara HHEM 正确加载后，增加 NLI 评分路径，与现有 LLM-as-judge 并行
7. **EvalOps 集成（orchestration 层）**：在 `run_naive_rag_baseline.py` / `main.py` 中上报版本化 `EvalRunReport`；同步更新下游分析脚本兼容新字段

### 2.2 质量提升目标（门控：Week 7 Day 11 前 MVP 已完成）

8. **SPLADE 混合检索**：引入 SPLADE（v2022 原版），与 dense 做 RRF 融合
9. **Setwise 两阶段 reranking**：cross-encoder 粗排 + LLM 精排组合

### 2.3 探索性目标（有余力则做）

10. **HippoRAG-lite 多跳检索**：基于 NER + PPR 的启发性实验（非论文复现，结果不保证）
11. **多生成模型对比**：分离检索贡献与生成贡献
12. **Agentic 迭代检索原型**：Plan-Retrieve-Evaluate-Retrieve-Generate 循环

---

## 3. 详细实现规划

### 3.0 前置任务：Document Schema 迁移（A0）

**背景**：当前 `Document` 只有 `doc_id / text / title` 三个字段（`src/types.py`），`docstore.py` 的 JSONL 序列化也只保存这三个字段。page grounding、section-aware retrieval 和 API response 的 `metadata` 字段均依赖扩展后的 schema。

**这是 A1/A2/A3 的强前置，必须最先完成。**

**改动方案**：

```python
# src/types.py - Document 扩展
@dataclass
class Document:
    doc_id: str
    text: str
    title: str
    # 新增，均有默认值保持向后兼容
    page_start: int | None = None   # 替代单一 page_number，支持跨页 chunk
    page_end: int | None = None     # 若 chunk 不跨页则 page_end == page_start
    section: str | None = None
    source: str | None = None
    extra_metadata: dict = field(default_factory=dict)
```

> **page_start/page_end 设计说明**：chunk 在 PDF 解析时可能跨越两页边界（如一段话从第 3 页延续到第 4 页），单个 `page_number` 无法表示。page grounding 评测使用 gold `page_set` ∩ `{page_start..page_end}` 是否为非空集合。

**docstore 向后兼容**：读取旧 JSONL 时，新字段默认为 `None`；写入新 JSONL 时输出完整字段。

**A0 完整文件改动范围**（比 v1.1 扩展，覆盖所有序列化路径）：

| 文件 | 操作 | 原因 |
|------|------|------|
| `src/types.py` | 扩展 `Document` dataclass | schema 定义 |
| `src/retrieval/docstore.py` | 更新序列化/反序列化，兼容旧格式 | 直接读写 Document |
| `src/corpus.py` | 更新 corpus reader，保留新字段 | 当前 `load_corpus()` 会丢弃 extra fields |
| `scripts/build_retrieval_indexes.py` | 更新 Document 序列化逻辑 | L167 硬编码只写 doc_id/title/text |
| `scripts/build_dense_sharded_index.py` | 同上 | L136 同样硬编码 |
| `tests/test_docstore_migration.py` | 新增：测试旧格式 JSONL 仍可正常加载 | 向后兼容验证 |

---

### 3.1 Phase A：MVP 转型（第 6-7 周）

#### A1. 文档 Ingestion Pipeline

**目标**：从"只能处理 FlashRAG 格式纯文本数据集"升级为"能处理真实 PDF 文档"

**MVP 范围**：仅支持 native-text PDF（`pdfplumber`）。OCR/扫描件支持为 stretch goal，仅在时间充裕时实施。

**实现路径**：

```
PDF 文件（native-text）
  ↓
[解析层] pdfplumber → 按页提取文本 + 页码范围
  ↓
[Chunking] 滑动窗口 + 段落边界感知，chunk_size=256 tokens，overlap=32 tokens
           每个 chunk 记录 page_start, page_end（支持跨页），section（启发式 heading 检测）
  ↓
[Metadata] doc_id（hash）, title, source, page_start, page_end, section, chunk_id
  ↓
[持久化] 写入 JSONL docstore（复用扩展后的 src/types.py Document）
  ↓
[异步索引] 触发 index build job（见 A2 的 /ingest 设计）
```

**Stretch（仅时间充裕时）**：

```
扫描件 PDF
  ↓
[OCR] surya-ocr / marker → page-level text
  ↓
（后续与 native-text 路径汇合）
```

**文件改动**：

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/ingestion/__init__.py` | 新增 | 模块入口 |
| `src/ingestion/pdf_parser.py` | 新增 | `PdfParser`：pdfplumber 解析，返回 `List[Document]` |
| `src/ingestion/ocr_parser.py` | 新增（stretch） | `OcrParser`：surya/marker 解析扫描件 |
| `src/ingestion/chunker.py` | 新增 | `TokenAwareChunker`：tiktoken 计算 chunk，保留 metadata |
| `src/ingestion/factory.py` | 新增 | `get_parser(mode: str)` 工厂函数 |
| `scripts/ingest_documents.py` | 新增 | CLI 入口 |
| `pyproject.toml` | 修改 | 新增：`pdfplumber>=0.11.0`；OCR 依赖仅按需安装 |

**验收标准**：
- 输入一个 10 页 native-text PDF，正确解析出带 `page_start / page_end` 的 chunks
- 跨页 chunk 的 `page_start != page_end`，单页 chunk 的 `page_start == page_end`
- Chunk 数量和 token 数量有日志输出
- 旧有 FlashRAG JSONL 格式仍可被 docstore 正常加载（向后兼容，A0 测试通过）

---

#### A2. 检索 API 化（FastAPI 服务）

**目标**：将检索能力封装为 HTTP 服务，供 agent runtime 调用

**前置要求**：需先引入 `ScoredDocument` 类型，因为现有检索层只返回 `List[Document]`，不携带 score。

**新增 ScoredDocument（Adapter 模式，不破坏现有代码路径）**：

```python
# src/types.py 新增
@dataclass
class ScoredDocument:
    document: Document
    score: float          # 归一化相似度或 rerank score
    retrieval_stage: str  # "dense" | "bm25" | "rerank"
```

**关键设计决策**：`ScoredDocument` 仅在 API 层（`src/api/`）使用，**不**修改现有 `pipeline.py` / `cross_encoder.py` / `hybrid.py` 的内部 `List[Document]` 接口。由 API handler 负责在最外层将检索结果包装为 `ScoredDocument`：

```python
# src/api/handlers.py 中的 adapter（示意）
def _wrap_scores(docs: list[Document], scores: list[float]) -> list[ScoredDocument]:
    return [ScoredDocument(doc, score, "dense") for doc, score in zip(docs, scores)]
```

这样 Phase 1 的 pipeline 全路径不需要动，仅 API 层新增包装逻辑。需要 score 的前提是让各 retriever 返回 `(docs, scores)` tuple（可选），或从 cross_encoder 直接获取 rerank score。

**受影响文件（仅 API 层，不动 pipeline 内部）**：

| 文件 | 操作 |
|------|------|
| `src/types.py` | 新增 `ScoredDocument` dataclass |
| `src/api/handlers.py` | 实现 `_wrap_scores` adapter |
| `src/reranking/cross_encoder.py` | 新增可选 `rerank_with_scores()` 返回 `(docs, scores)` |

**接口设计**：

```
POST /v1/ingest
  # 接受文件内容（multipart/form-data）或 base64 编码
  # 注意：file_path 方式仅适用于 demo/单机场景，已知安全限制
  Request:  multipart: file=<binary>, parser=pdf|ocr, index_name=<str>
  Response: { "job_id": "uuid", "status": "queued" }
  # ingest 是异步 job，不阻塞返回

GET /v1/ingest/{job_id}
  Response: { "job_id": "...", "status": "running|done|failed",
              "chunks_indexed": 128, "index_id": "..." }

POST /v1/retrieve
  Request:  {
              "query": "...",
              "index_id": "...",
              "top_k": 5,
              "retrieval_profile": "dense_rerank_v1",  # 对应 config hash 或命名配置
              "use_reranker": true
            }
  Response: {
              "results": [
                {
                  "doc_id": "...",
                  "text": "...",
                  "score": 0.87,
                  "metadata": { "source": "...", "page": 3, "section": "..." }
                }
              ],
              "latency_ms": 142,
              "retrieval_profile": "dense_rerank_v1",  # echo back，便于结果追溯
              "index_id": "..."
            }

GET /v1/health
  Response: { "status": "ok", "indexes_loaded": ["wiki18_21m", "my_pdf_index"] }
```

> **设计说明**：
> - `/ingest` 改为 multipart 上传（demo 场景可保留 `file_path` 模式但标注为 dev-only）
> - `retrieval_profile` 参数接受命名配置或 config hash，响应中 echo back，确保结果可追溯
> - `index_registry` 为单进程单例，内含加载锁（`threading.Lock`）防止并发初始化竞争；demo 场景不做多进程扩展

**LazyDocstore 并发问题修复**（A2 的前置实现要求）：

当前 `src/retrieval/docstore.py` 的 `LazyDocstore` 持有共享文件句柄，`seek()`/`read()` 是非线程安全操作。在多线程 FastAPI 下，并发请求会引起文件指针竞争，返回乱序或损坏的 Document 行。

修复方案（二选一，优先方案 A）：

```python
# 方案 A（推荐）：per-request open，无需锁
def get_by_offset(self, offset: int) -> Document:
    with open(self._path, "r") as f:   # 每次请求单独打开
        f.seek(offset)
        return Document(**json.loads(f.readline()))

# 方案 B（备选）：保留共享句柄，加 threading.Lock
def get_by_offset(self, offset: int) -> Document:
    with self._lock:
        self._file.seek(offset)
        return Document(**json.loads(self._file.readline()))
```

方案 A 稍慢（每次系统调用），但对 demo 规模并发完全够用，无死锁风险。

**A2 的 /ingest 规范补全**：

`/ingest` 的核心职责是：接收文件 → 解析 chunks → 持久化到 docstore JSONL → 构建索引 → 生成 manifest.json → 注册到 `index_registry`。

```
index_id = hash(index_name + timestamp)[:8]
manifest 路径 = data/indexes/{index_id}/manifest.json
manifest 结构 = {
  "index_id": "...",
  "index_name": "...",
  "created_at": "...",
  "num_chunks": 128,
  "docstore_path": "data/indexes/{index_id}/docstore.jsonl",
  "faiss_path": "data/indexes/{index_id}/index.faiss",   # dense 时
  "bm25_path": "data/indexes/{index_id}/bm25.pkl",       # bm25 时
}
```

**策略**：MVP 阶段仅支持全量重建（不支持 append），这样 ingest job 逻辑最简单。Append 模式列为 known limitation。

**文件改动**：

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/types.py` | 修改 | 新增 `ScoredDocument` dataclass |
| `src/retrieval/docstore.py` | 修改 | 修复并发 bug（方案 A：per-request open） |
| `src/api/__init__.py` | 新增 | 模块入口 |
| `src/api/server.py` | 新增 | FastAPI app 实例，路由注册 |
| `src/api/models.py` | 新增 | Pydantic request/response 模型 |
| `src/api/handlers.py` | 新增 | `/ingest`, `/retrieve`, `/health` 处理器 + `_wrap_scores` adapter |
| `src/api/index_registry.py` | 新增 | 多 index 管理（单例 + 加载锁），基于 manifest.json 发现 |
| `src/api/ingest_worker.py` | 新增 | 同步 ingest + build job（thread pool executor，全量重建） |
| `scripts/start_api.py` | 新增 | `uvicorn src.api.server:app --host 0.0.0.0 --port 8080` |
| `pyproject.toml` | 修改 | 新增：`fastapi>=0.115.0`, `uvicorn[standard]>=0.32.0` |

**MVP 范围限制**（Week 7 内可完成）：
- `/retrieve`：read-only，基于**预先构建好的 index**（不在 API 内 build），Week 7 先交付此接口
- `/ingest`：async job + manifest + 全量 rebuild，Week 7 后半段若时间允许再实现；否则作为 stretch

**验收标准**：
- `POST /v1/retrieve` 返回带 `score` 和 `metadata.page_start/page_end` 的正确格式
- 单线程下 p99 延迟 < 3s（dense 检索，无 rerank，toy index）
- `retrieval_profile` 字段被正确 echo back
- 并发 2 请求不出现 docstore 竞争问题（修复 LazyDocstore 后）

---

#### A3. Citation / Page Grounding 评测

**目标**：新增面向真实文档场景的评测维度，与现有 EM/F1 **并列使用**

> **重要澄清（v1.2 修正）**：citation 指标**不是**从 LLM 输出中提取引用标记。当前生成器（`src/generation/openai_compatible.py` L16/L129）的 system prompt 要求模型只输出最终答案文本，不输出 passage 引用。因此，所有 citation 指标均为 **post-hoc NLI attribution**：对每个已检索 passage，用 NLI 模型判断它是否 entails 生成的答案句，从而反推归因关系。这不需要修改 LLM prompt，但也意味着它度量的是"支持关系"而非"LLM 实际使用了哪个 passage"。

> **数据可用性**：citation/page grounding 指标仅适用于新建的 PDF Q&A 数据集（有 gold passage + page_start/page_end 标注）。HotpotQA/NQ 仅有 gold titles，不适用。

**新增指标及操作性定义**：

| 指标 | 定义 | 计算方式 | 说明 |
|------|------|----------|------|
| `answer_attribution_rate` | 答案中每句话是否有至少一个检索 passage 能 entail 它 | 句粒度 NLI：每句 entailment prob ≥ 0.5 则视为已归因；归因句数 / 总句数 | 主要指标 |
| `supporting_passage_hit` | 被 NLI 判为支持答案的 passage 占检索 top-k 的比例 | entailment prob ≥ 0.5 的 passage 数 / k | 衡量检索精度 |
| `page_grounding_accuracy` | 被判为 supporting 的 passage 的 page 范围是否与 gold page_set 有交集 | gold_page_set ∩ {page_start..page_end} 非空 | 仅 PDF 数据集 |

**NLI 模型选型与正确加载方式**：

使用 Vectara HHEM（`vectara/hallucination_evaluation_model`），该模型为 sequence-classification 架构，**不是** `CrossEncoder`：

```python
# 正确加载方式（需 trust_remote_code=True）
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class HHEMScorer:
    def __init__(self):
        model_name = "vectara/hallucination_evaluation_model"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()

    def score(self, premise: str, hypothesis: str) -> float:
        """返回 entailment probability（0=幻觉，1=忠实）"""
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        # 标签顺序：0=entailment, 1=hallucination（需验证 model card）
        probs = torch.softmax(logits, dim=-1)
        return probs[0][0].item()  # entailment prob
```

> **注意**：部署前需在本机验证标签顺序（`model.config.id2label`）与 score 方向是否与 model card 一致。

**评分规范（操作性定义补充）**：
- 句子分割：使用 `nltk.sent_tokenize`，英文场景
- NLI 阈值：entailment prob ≥ 0.5 视为支持
- 聚合方式：macro average（每条 query 算一个 precision，再对所有 query 平均）
- 缺少 gold passage 的样本标记为 `null`，不纳入均值计算

**忠实度评分的补充**：NLI-based 路径与现有 LLM-as-judge 并行保留，对比两者分布差异作为本次评测的额外发现。

**文件改动**：

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/evaluation/hhem_scorer.py` | 新增 | `HHEMScorer`：正确加载 Vectara HHEM |
| `src/evaluation/citation.py` | 新增 | `CitationEvaluator`：计算 `answer_attribution_rate` / `supporting_passage_hit` / `page_grounding_accuracy` |
| `src/evaluation/faithfulness.py` | 修改 | 增加 NLI-based 路径，保留 LLM-as-judge |
| `src/types.py` | 修改 | `RunExampleResult` 新增字段（均有默认 None） |
| `src/pipeline.py` | 修改 | 在评测阶段调用 `CitationEvaluator`（仅 PDF 场景） |
| `scripts/score_citation.py` | 新增 | 批量对已有 run 结果补打 citation 分 |

**验收标准**：
- HHEM 模型正确加载（无 `trust_remote_code` 错误）
- 对手工标注的 PDF Q&A 数据集（≥ 100 句子-passage 对），NLI 评分与人工判断的 Cohen's Kappa ≥ 0.6
- `page_grounding_accuracy` 使用 ±1 页容差，而非精确页码匹配

---

### 3.2 Phase B：检索质量提升（Gate：Week 7 Day 11 前 MVP 已完成）

> **执行条件**：B1/B2 仅在 A0-A3 全部验收通过后启动。否则时间优先保障 C1/C2（EvalOps 集成）。

#### B1. SPLADE 混合检索

**背景**：当前 `normalization_or_alias_suspect` 失败桶有 39/200 条，SPLADE 的 learned sparse representation 对词汇变体类问题有专项优势。

**实现方案**：使用 `naver-splade/splade-cocondenser-ensembledistil`

```python
# src/retrieval/splade.py
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch, scipy.sparse

class SPLADERetriever:
    def __init__(self, model_name="naver-splade/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def encode(self, text: str) -> dict[int, float]:
        """返回 {vocab_token_id: weight} 稀疏向量"""
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model(**inputs)
        # SPLADE: max-pooling over sequence, then log(1+relu(logits))
        vec = torch.log1p(torch.relu(output.logits)).max(dim=1).values.squeeze()
        indices = vec.nonzero(as_tuple=True)[0].tolist()
        return {i: vec[i].item() for i in indices}

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        # 基于预建 inverted index 做 dot product 检索
        # 注意：返回裸 Document 列表，与现有检索层接口一致
        # ScoredDocument 包装由 API 层 _wrap_scores() 负责
        ...
```

**与 Dense 融合（RRF）**：

```python
# src/retrieval/hybrid.py 扩展
def rrf_fusion(dense_results: list[Document],
               splade_results: list[Document],
               k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion"""
    ...
```

**实验设计**：

| 配置 | 目的 |
|------|------|
| Dense Only (baseline) | Phase 1 基线复现 |
| SPLADE Only | 验证 alias 类 query 的改善 |
| Dense + SPLADE (RRF) | 预期 overall recall 提升 |

**Research basis**：
- SPLADE++（Formal et al., SIGIR 2022）在 BEIR 评测集上 in/out-domain 均优于纯 dense
- SPLATE（SIGIR 2024）提出从 ColBERT 映射到稀疏表示，在 50 个候选文档上检索延迟 < 10ms

> **注**：SPLATE 的 <10ms 延迟数据针对的是"从 50 个候选中再检索"的特定架构，不能直接作为本项目全量索引检索的延迟预期。本项目需实测。

**文件改动**：

| 文件 | 操作 |
|------|------|
| `src/retrieval/splade.py` | 新增 SPLADE 检索器 |
| `src/retrieval/hybrid.py` | 扩展支持 dense+splade RRF |
| `src/retrieval/factory.py` | 注册 `splade` 和 `dense_splade_hybrid` 模式 |
| `scripts/build_splade_index.py` | 新增 inverted index 构建脚本 |
| `config/phase5/C7_dense_splade_hybrid.yaml` | 新增实验配置 |

---

#### B2. 两阶段 Reranking（Setwise LLM 精排）

**背景**：改善 `both_gold_after_dedup_but_lost_after_rerank` 失败桶。

**两阶段设计**：

```
Retrieval → top-100 passages
  ↓ Stage 1: Cross-encoder (ms-marco-MiniLM-L-6-v2) 粗排
top-20 passages
  ↓ Stage 2: Setwise LLM reranking（Qwen2.5-3B-Instruct 本地推理）精排
top-5 passages → Generation
```

**Setwise Prompt 设计**：

```
Given the query: "{query}"

Rank the following passages from most to least relevant.
Output ONLY a comma-separated list of passage indices (e.g., "3,1,5,2,4"):

[1] {passage_1}
[2] {passage_2}
...
[N] {passage_N}
```

> 候选集限制：Stage 2 输入 ≤ 10 条，避免 context 过长导致 3B 模型幻觉排序。

**文件改动**：

| 文件 | 操作 |
|------|------|
| `src/reranking/setwise.py` | 新增 `SetwiseLLMReranker` |
| `src/reranking/pipeline.py` | 新增两阶段 rerank 流程编排 |
| `src/pipeline.py` | 支持 `reranker_mode: two_stage` 配置 |
| `config/phase5/C6_two_stage_rerank.yaml` | 新增实验配置 |

---

### 3.3 Phase C：EvalOps 集成（Week 7 后半段）

#### C1. 标准化 Metric 上报

**统一 Schema（版本化）**：

```python
@dataclass
class EvalRunReport:
    # Schema 版本，便于 EvalOps 侧做兼容处理
    schema_version: str = "rag/v1"

    # 标识信息
    run_id: str           # UUID
    app_type: str         # "rag"
    timestamp: str        # ISO 8601

    # 版本信息
    dataset_name: str     # "hotpotqa" | "nq" | "triviaqa" | "custom_pdf"
    dataset_version: str  # "flashrag_v1" | "local_20260404"
    config_version: str   # git commit hash 或 config 文件 sha256[:8]
    retrieval_profile: str # 与 API /retrieve 中的同名参数对应
    model_name: str       # "MiniMax-M2.5" | "Qwen2.5-7B"

    # 核心 metrics
    num_queries: int
    answer_f1: float
    answer_em: float
    recall_at_k: float
    answer_attribution_rate: float | None  # post-hoc NLI，仅 PDF 数据集有效
    supporting_passage_hit: float | None   # top-k 中 NLI 支持答案的 passage 比例
    page_grounding_accuracy: float | None  # gold_page_set ∩ chunk page range 非空比例
    faithfulness_nli: float | None         # NLI-based（HHEM）
    faithfulness_llm: float | None         # LLM-as-judge（保留，并行对比）
    hallucination_rate: float

    # 效率 metrics
    avg_latency_ms: float
    avg_cost_usd: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float

    # 状态
    generation_error_rate: float
    config_path: str
    artifact_dir: str
```

**文件改动**：

**EvalOps 上报层归属修正（v1.2）**：

`run_naive_rag()` 函数（`src/pipeline.py`）仅返回 `(results, metrics)`，不持有 run_id / artifact_dir / dataset_name 等上下文。这些元数据由 orchestration 层管理：`main.py` 和 `scripts/run_naive_rag_baseline.py`（L775 处持久化结果）。

因此 EvalOpsClient 调用应放在**这两个脚本的 run 结束后**，而非 pipeline 内部。

**另需兼容**：新增字段（`attribution_rate`、`page_grounding` 等）会改变 `predictions.json` / `metrics.json` 的结构，下游脚本需一并更新：

| 文件 | 操作 |
|------|------|
| `src/evalops/__init__.py` | 新增模块 |
| `src/evalops/client.py` | 新增：`EvalOpsClient`，失败静默跳过 |
| `src/evalops/schema.py` | 新增：`EvalRunReport` dataclass（含 `schema_version: "rag/v1"`） |
| `src/evalops/adapter.py` | 新增：从 `(results, metrics, run_meta)` 转换为 `EvalRunReport` |
| `scripts/run_naive_rag_baseline.py` | 修改：L775 之后调用 `EvalOpsClient.report()` |
| `main.py` | 修改：toy 场景接入 EvalOps |
| `scripts/aggregate_experiment_results.py` | 修改：兼容新增字段（加 `get(key, None)` 防止 KeyError） |
| `scripts/export_dashboard_data.py` | 修改：dashboard bundle 支持新指标字段 |
| `scripts/score_faithfulness.py` | 修改：输出路径和字段命名与新 schema 对齐 |
| `.env.example` | 修改：新增 `EVALOPS_API_URL` / `EVALOPS_API_KEY` |

---

#### C2. Bad Case 可追溯性

每个失败样本的 `RunExampleResult` 新增字段：

```python
failure_stage: str      # "retrieval_miss" | "rerank_drop" | "generation_fail" | "citation_miss"
failure_detail: str     # 人可读的失败原因
run_id: str             # 与 EvalRunReport.run_id 一致，便于在 EvalOps 中关联
```

---

### 3.4 Phase D：探索性改进（有余力则做）

#### D1. HippoRAG-lite 多跳检索（启发性实验）

> **注意**：此实现为受 HippoRAG 2（arXiv:2502.14802）启发的轻量启发式实验，不是论文复现。HippoRAG 2 的 +9.5% F1 结论基于其完整框架，本项目的简化版本能否继承类似增益尚不确定，应将结果作为探索性数据点而非预期目标。

**轻量实现思路**：

```
Step 1: 对 passages 做 NER（使用 spacy en_core_web_sm）
Step 2: 构建 passage-entity 二部图
Step 3: Query → 第一跳 dense 检索 → seed passages
Step 4: 从 seed passages 出发，Personalized PageRank 游走 → 相关 passages
Step 5: 合并第一跳 + PPR 结果，送 reranker
```

**评估方式**：与 Phase 1 基线对比 `RecallAllGold@k_title`，分析改善的 query 类型。

#### D2. 多生成模型对比实验

**目标**：分离"检索质量"与"生成能力"对最终 F1 的贡献（固定检索配置，只换 LLM）

| 模型 | 类型 | 预期成本 |
|------|------|----------|
| MiniMax-M2.5（当前） | 商业 API | $0.0017/query |
| Qwen2.5-7B-Instruct | 本地推理 | ~$0（GPU 电费） |
| GPT-4o-mini | 商业 API | ~$0.0005/query |

**核心问题**：当前 F1 只有 0.21、Hallucination Rate 88-97%，是模型能力问题还是 prompt 问题？

#### D3. Agentic 迭代检索原型

**架构参考**：PAR2-RAG（arXiv:2603.29085）Plan-Act-Review 框架

```python
def agentic_retrieve(query: str, max_hops: int = 3) -> list[Document]:
    plan = llm_decompose(query)
    all_docs = []
    for sub_q in plan.sub_questions:
        docs = retriever.retrieve(sub_q)
        gap = llm_evaluate_gap(sub_q, docs)
        if gap.needs_more:
            docs += retriever.retrieve(gap.follow_up_query)
        all_docs.extend(docs)
    return dedup_and_rerank(all_docs)
```

**与 agent runtime 集成**：`llm-coding-agent-system` 调用 `/v1/retrieve` 时传 `retrieval_profile=agentic_v1`，触发多跳迭代。

---

## 4. 新增依赖汇总

```toml
# pyproject.toml 新增
dependencies = [
    # 文档解析（MVP）
    "pdfplumber>=0.11.0",
    # OCR（stretch，按需安装）
    # "surya-ocr>=0.6.0",

    # API 服务
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.7.0",

    # SPLADE 检索（B1，Gate 后启用）
    "transformers>=4.40.0",

    # NLI-based faithfulness
    # vectara/hallucination_evaluation_model 通过 transformers 加载
    # 需要 trust_remote_code=True，无需额外依赖

    # EvalOps 集成
    "httpx>=0.27.0",

    # 句子分割（citation 评测用）
    "nltk>=3.9.0",

    # 图检索（D1 探索，可选）
    # "spacy>=3.7.0",
    # "networkx>=3.3",
]
```

---

## 5. 目录结构变更

```diff
rag-benchmark-system/
├── src/
│   ├── pipeline.py              (修改：仅 PDF 场景调用 CitationEvaluator；EvalOps 上报已移入 orchestration 层)
+  ├── types.py                  (修改：Document 扩展 page_start/page_end + ScoredDocument 新增)
+  ├── corpus.py                 (修改：A0 - load_corpus() 保留新字段)
│   ├── retrieval/
│   │   ├── splade.py            ← 新增（B1，Gate 后）
│   │   ├── hybrid.py            (修改：支持 dense+splade RRF，内部仍用 List[Document]）
│   │   └── docstore.py          (修改：向后兼容新 Document schema + 并发 bug 修复)
│   ├── reranking/
│   │   ├── cross_encoder.py     (修改：新增可选 rerank_with_scores() 返回 (docs, scores))
│   │   └── setwise.py           ← 新增（B2，Gate 后）
+  ├── ingestion/               ← 新增模块（Week 6）
+  │   ├── __init__.py
+  │   ├── pdf_parser.py
+  │   ├── ocr_parser.py         （stretch）
+  │   ├── chunker.py
+  │   └── factory.py
+  ├── api/                     ← 新增模块（Week 7）
+  │   ├── __init__.py
+  │   ├── server.py
+  │   ├── models.py
+  │   ├── handlers.py
+  │   ├── index_registry.py
+  │   └── ingest_worker.py
+  ├── evalops/                 ← 新增模块（Week 7）
+  │   ├── __init__.py
+  │   ├── client.py
+  │   ├── schema.py
+  │   └── adapter.py
│   └── evaluation/
│       ├── metrics.py           (保持)
│       ├── faithfulness.py      (修改：加 NLI 路径)
+       ├── hhem_scorer.py        ← 新增（正确加载 Vectara HHEM）
+       └── citation.py          ← 新增
├── scripts/
+   ├── ingest_documents.py      ← 新增
+   ├── start_api.py             ← 新增
+   ├── build_splade_index.py    ← 新增（B1，Gate 后）
+   └── score_citation.py        ← 新增
├── config/
+   └── phase5/                  ← 新增
+       ├── C6_two_stage_rerank.yaml
+       ├── C7_dense_splade_hybrid.yaml
+       └── C8_agentic_retrieve.yaml
└── tests/
+   ├── test_docstore_migration.py ← 新增（A0 验收）
+   ├── test_pdf_ingestion.py    ← 新增
+   ├── test_api_retrieve.py     ← 新增
+   ├── test_citation_eval.py    ← 新增
+   └── test_splade_retriever.py ← 新增（B1，Gate 后）
```

---

## 6. 实验计划（Phase 5 Matrix）

### 6.1 实验矩阵

| Config | 检索 | Reranker | 查询扩展 | 主要测试点 |
|--------|------|----------|----------|-----------|
| C1 (baseline) | dense_sharded | cross-encoder | off | Phase 1 最优基线复现 |
| C6 | dense_sharded | cross-encoder → setwise | off | 两阶段 rerank 增益 |
| C7 | dense+splade (RRF) | cross-encoder | off | SPLADE 混合检索增益 |
| C8 | dense+splade (RRF) | cross-encoder → setwise | decompose | 组合最优配置 |
| C9 | dense (PDF index) | cross-encoder | off | 真实 PDF 文档场景 |

> C6-C8 仅在 B1/B2 Gate 通过时执行

### 6.2 数据集

- HotpotQA（保持 200q，对比 Phase 1）
- NQ（保持 200q）
- **新增**：自定义 PDF 数据集
  - 20-50 个真实 PDF 文档（技术报告/学术论文）
  - ≥ 100 条 Q&A 对 + gold passage + gold page_set 标注
  - 标注流程：GPT-4o 辅助生成初稿，人工审核全部，保证 citation/page 字段可用

### 6.3 核心评测指标（Phase 5）

| 指标 | Phase 1 基线 | Phase 2 目标 | 数据集 | 说明 |
|------|-------------|-------------|--------|------|
| RecallAllGold@k_title | 0.235 | ≥ 0.27 | HotpotQA | B1/B2 Gate 通过后才可能达到 |
| Answer F1 | 0.126 | ≥ 0.15 | NQ | 同上 |
| Answer EM | 0.045 | ≥ 0.06 | NQ | 同上 |
| Answer Attribution Rate | N/A | ≥ 0.65 | PDF | post-hoc NLI，entail ≥ 0.5 的句子比例 |
| Supporting Passage Hit | N/A | ≥ 0.60 | PDF | top-k 中 NLI 支持答案的 passage 比例 |
| Page Grounding Accuracy | N/A | ≥ 0.70 | PDF | gold_page_set ∩ {page_start..page_end} 非空 |
| HHEM vs 人工 Cohen's Kappa | N/A | ≥ 0.6 | PDF（50+ pairs 可行性） | 100+ pairs 时更可信 |
| API Latency（单线程） | N/A | < 3s | toy index，无 rerank | 并发后续再测 |

---

## 7. 研究依据

| 改进点 | 论文 / 来源 | 关键数据 | 准确性说明 |
|--------|-------------|----------|-----------|
| SPLADE 混合检索 | SPLADE++（Formal et al., **SIGIR 2022**） | BEIR 上优于纯 dense | 年份勿误写为 2025 |
| SPLATE 架构 | SPLATE（SIGIR 2024） | 稀疏候选检索架构 | <10ms 针对特定 50 候选架构，不直接适用于本项目 |
| Setwise LLM reranking | Zhuang et al., 2023 Setwise | 效率优于 listwise | 需实测本项目 latency |
| NLI-based faithfulness | HHEM（Vectara, 2023），RAGTruth（IEEE S&P 2024） | judge 间分歧研究 | FaithBench 为摘要 benchmark，不直接引用 |
| Citation grounding | RAGAS（arXiv:2309.15217），DeepEval（2025） | Citation 指标标准化 | |
| HippoRAG 2 多跳（启发性） | arXiv:2502.14802（2025-02） | 2WikiMultiHop +9.5% F1（完整框架） | 本项目 lite 版收益不保证 |
| Query decomposition | arXiv:2507.00355 | HotpotQA MRR@10 +36.7% | |
| Agentic iterative retrieval | PAR2-RAG（arXiv:2603.29085，2026） | 多跳 benchmark SOTA | |
| Two-stage reranking | "Evolution of Reranking"（arXiv:2512.16236） | cross-encoder + LLM 组合 | |

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|----------|
| **A0 迁移漏掉 corpus.py / build 脚本，page 字段在索引阶段丢失** | 高 | 高 | A0 文件清单（v1.2）已包含全部 5 个文件；迁移后用 toy PDF 做端到端验证，检查 page_start 是否出现在检索结果中 |
| **Document schema 迁移破坏现有 run 加载** | 中 | 高 | `test_docstore_migration.py` 必须覆盖旧格式加载；迁移前备份现有 JSONL |
| **LazyDocstore 并发 bug 导致 API 返回乱行** | 中 | 高 | 采用方案 A（per-request open），修复后并发 2 请求压测验证 |
| **Vectara HHEM 模型 trust_remote_code 加载失败** | 中 | 中 | 先验证本机加载 + 标签顺序；备选：`cross-encoder/nli-deberta-v3-base` |
| **下游脚本（aggregate / export / score_faithfulness）因新字段 KeyError 崩溃** | 中 | 中 | 所有字段访问改为 `.get(key, None)`；C2 阶段统一修复 |
| surya/marker OCR 在 CPU 上速度过慢 | 高 | 低 | MVP 仅做 native-text PDF；OCR 为 stretch，可完全跳过 |
| SPLADE 索引构建内存不足（21M passages） | 中 | 高 | Gate 通过后先在 PDF index 上验证；21M 仅在 32GB+ 内存机器启用 |
| EvalOps Platform 接口未就绪 | 中 | 低 | Client mock 模式；本地 run 不依赖 EvalOps 可用性 |
| Setwise LLM 推理延迟超标 | 中 | 中 | 候选集 ≤ 10 条；3B 模型；仅离线评测启用 |
| PDF Q&A 标注工作量超出 Week 6 时间 | 高 | 中 | 标注为 stretch；HHEM 验证可用 50 条进行可行性检查，kappa 验证推迟到有 100 条时 |
| B1/B2 挤占 MVP 时间 | 高 | 高 | 严格执行 Gate：Day 11 前 MVP 未完成则跳过 B1/B2 |

---

## 9. 与整体路线图的对接关系

```
llm-coding-agent-system
  └── POST /v1/retrieve（retrieval_profile 参数）
       获得 ScoredDocument 列表，包含 page/section metadata
  └── 评测结果上报到 llm-evalops-platform

llm-evalops-platform
  └── 接收 EvalRunReport（schema_version: "rag/v1"）
  └── 展示 citation / faithfulness_nli / latency / cost dashboard
  └── release gate：answer_attribution_rate >= 0.65 才能 promote 配置

rag-benchmark-system（本项目）
  └── 提供知识检索 API（/v1/retrieve，携带 retrieval_profile）
  └── 提供文档 ingestion（/v1/ingest 异步 job）
  └── 产出 citation-aware 评测结果（接入 EvalOps，schema 版本化）
```

---

## 10. 面试叙事目标

Phase 2 完成后，这个项目的简历表达应该升级为：

> **Before（Phase 1）**：Systematically benchmarked 5 RAG configurations across 3 QA datasets with detailed failure taxonomy and cost analysis.

> **After（Phase 2）**：Extended a RAG benchmark system into an agent-ready document intelligence subsystem with PDF ingestion, a scored read-only retrieval API with retrieval profiling, post-hoc NLI attribution evaluation alongside existing LLM judges, and hybrid SPLADE+dense retrieval — integrated into a shared EvalOps platform at the orchestration layer with versioned run schemas, retrieval profile tracing, and release gates.

这套表达覆盖了路线图要求的所有核心信号：
- **retrieval for agent workflows**（retrieval_profile API）✓
- **complex document processing**（PDF ingestion，page_start/page_end）✓
- **citation grounding**（post-hoc NLI attribution，operationally defined）✓
- **async ingestion**（/ingest job，stretch 目标；MVP 为 read-only /retrieve）△
- **EvalOps integration**（orchestration 层 versioned schema + profile tracing）✓

---

## 附录 A：Phase 1 关键数据存档

| 指标 | 值 | 配置 |
|------|----|----|
| RecallAllGold@k_title | 0.235 | dense_sharded, top-100, rerank, no expansion |
| RecallAllGold@k_title (coverage-filtered) | 0.3919 | 同上 |
| Answer F1 (HotpotQA) | 0.121 | C2 (Dense+Rerank) |
| Answer F1 (NQ) | 0.126 | C5 (Rerank+Decompose) |
| Answer F1 (TriviaQA) | 0.212 | C2 (Dense+Rerank) |
| 主要失败桶 | query_formulation_gap: 64/200 | 基于 title BM25 probing 后分析 |
| 生成 Error Rate | 0.08 | closure-safe 配置 |

---

## 附录 B：参考论文列表

1. HippoRAG 2 — arXiv:2502.14802 — https://arxiv.org/abs/2502.14802
2. PAR2-RAG — arXiv:2603.29085 — https://arxiv.org/html/2603.29085
3. A-RAG — arXiv:2602.03442 — https://arxiv.org/html/2602.03442v1
4. Agentic RAG Survey — arXiv:2501.09136 — https://arxiv.org/abs/2501.09136
5. Query Decomposition for RAG — arXiv:2507.00355 — https://arxiv.org/abs/2507.00355
6. RAGTruth（Hallucination Benchmark）— IEEE S&P 2024
7. RAG Eval Survey — arXiv:2504.14891 — https://arxiv.org/html/2504.14891v1
8. SPLADE++（Formal et al.）— SIGIR 2022
9. SPLATE — SIGIR 2024 — https://dl.acm.org/doi/10.1145/3626772.3657968
10. Expanded-SPLADE — arXiv:2511.22263 — https://arxiv.org/html/2511.22263v1
11. Reranking Evolution — arXiv:2512.16236 — https://arxiv.org/html/2512.16236v1
12. SEAL-RAG — arXiv:2512.10787 — https://arxiv.org/pdf/2512.10787
13. HopRAG — arXiv:2502.12442 — https://arxiv.org/html/2502.12442v1
14. Credible Plan-Driven RAG — arXiv:2504.16787 — https://arxiv.org/pdf/2504.16787
15. Towards Practical GraphRAG — arXiv:2507.03226 — https://arxiv.org/abs/2507.03226
16. RAGAS — arXiv:2309.15217 — https://arxiv.org/abs/2309.15217
