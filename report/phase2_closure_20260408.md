# Phase 2 Closure Report: Agent Knowledge Subsystem

**日期**：2026-04-08  
**状态**：Phase 2 MVP 已完整交付，正式关闭

---

## 1. 背景

Phase 1（2026-03-10 关闭）的核心结论：检索质量已够用（Recall@k 0.63–0.81），生成才是真正瓶颈（F1 仅 0.10–0.21，幻觉率 88–97%）。继续在 HotpotQA recall 上做 ablation 边际收益极低。

Phase 2 目标：**转型为可被 agent runtime 调用的知识检索子系统**，新增真实文档处理能力、HTTP 服务接口、NLI 忠实度评测。

---

## 2. 交付清单

### A0 — Document Schema 迁移（强前置）

| 文件 | 改动 |
|------|------|
| `src/types.py` | `Document` 新增 `page_start/page_end/section/source/extra_metadata`（均有默认值，向后兼容）；新增 `ScoredDocument` API adapter；`RunExampleResult` 新增三个 NLI 字段 |
| `src/retrieval/docstore.py` | 序列化/反序列化更新为条件输出（None 字段不写入）；`LazyDocstore` 改为 per-request open，修复并发 bug；新增私有 `_doc_to_row` / `_row_to_doc` 辅助函数 |
| `src/corpus.py` | `_row_to_document()` 透传新字段 |
| `scripts/build_retrieval_indexes.py` | 序列化路径更新 |
| `scripts/build_dense_sharded_index.py` | `_write_doc_row()` 更新 |
| `tests/test_docstore_migration.py` | 新增 7 个测试：round-trip、向后兼容、LazyDocstore 并发安全、ScoredDocument |

**已知修复**：`extra_metadata: dict` 在 `frozen=True` dataclass 中会导致 `hash()` 抛 `TypeError`，已通过 `field(hash=False)` 修复。

### A1 — PDF Ingestion Pipeline

| 文件 | 内容 |
|------|------|
| `src/ingestion/pdf_parser.py` | `PdfParser`：pdfplumber 解析 native-text PDF，每页产出一个 Document，携带 `page_start/page_end/source` |
| `src/ingestion/chunker.py` | `TokenAwareChunker`：tiktoken 滑动窗口分块，chunk_size=256 tokens，overlap=32；跨页块保留父页码；`extra_metadata["chunk_index"]` |
| `src/ingestion/factory.py` | `get_parser(mode)` 工厂 |
| `scripts/ingest_documents.py` | CLI：`--input <pdf>... --output <jsonl> --chunk-size --overlap` |

**已知限制**：仅支持 native-text PDF。OCR/扫描件为 stretch goal，未实现。

### A2 — FastAPI 检索服务

| 文件 | 内容 |
|------|------|
| `src/api/models.py` | Pydantic 请求/响应模型：`RetrieveRequest`、`RetrievedPassage`（含 page 元数据）、`RetrieveResponse`、`HealthResponse` |
| `src/api/index_registry.py` | 线程安全多 index 注册（`threading.Lock`）；`load_from_config()` 从 YAML 加载 retriever |
| `src/api/handlers.py` | `handle_retrieve()` + `_wrap_scores()` adapter（rank-normalized surrogate score） |
| `src/api/server.py` | FastAPI app；lifespan 通过 `INDEX_CONFIG_<ID>` 环境变量批量加载 index |
| `scripts/start_api.py` | uvicorn 启动器，默认 `workers=1`（FAISS 不 fork-safe） |

**端点**：
- `POST /v1/retrieve` — 检索，返回带 page 元数据的 passage 列表
- `GET /v1/health` — 健康检查，列出已加载的 index id

**已知限制**：`/ingest` async job（文件上传 → build index）为 stretch goal，未实现。当前需要用 `scripts/ingest_documents.py` + `scripts/build_retrieval_indexes.py` 离线建好 index 再注册。

### A3 — NLI Citation 评测

| 文件 | 内容 |
|------|------|
| `src/evaluation/hhem_scorer.py` | `HHEMScorer`：加载 `vectara/hallucination_evaluation_model`（trust_remote_code=True，lazy import） |
| `src/evaluation/citation.py` | `CitationEvaluator`：计算 `answer_attribution_rate`、`supporting_passage_hit`、`page_grounding_accuracy` |
| `src/evaluation/faithfulness.py` | 追加 `nli_score_faithfulness()`，与原 LLM-as-judge `score_faithfulness()` 并行保留 |
| `scripts/score_citation.py` | 批量脚本：对已有 run results JSON 补打 NLI 分数 |

**已知限制**：`page_grounding_accuracy` 仅在 Document 有 `page_start` 元数据时有意义（即 PDF 来源），HotpotQA/NQ 旧数据该字段为 `None`。

---

## 3. 测试覆盖

| 测试文件 | 用例数 | 覆盖内容 |
|----------|--------|---------|
| `tests/test_docstore_migration.py` | 7 | A0 schema round-trip / 向后兼容 / 并发安全 / ScoredDocument |
| `tests/test_smoke_phase2.py` | 26 | A0/A1/A2/A3 全流程冒烟，含 FastAPI TestClient、in-memory PDF 解析 |
| `tests/test_*.py`（Phase 1 原有）| 77 | 全部通过，无退化 |

**预存在失败**（与 Phase 2 无关）：`tests/test_query_factory.py` 2 个测试因缺少 `LLM_BASE_URL` env var 而失败，Phase 2 前已存在。

---

## 4. Phase 2 评测结果

运行 `scripts/eval_phase2_full.py`（overlap-stub scorer，无需大模型下载）：

| 维度 | 指标 | 结果 |
|------|------|------|
| A0 向后兼容 | 旧格式 JSONL 加载成功率 | **100%** |
| A0 字段保留 | 新字段写入后读回完整率 | **100%** |
| A0 并发安全 | LazyDocstore 8 线程随机读准确率 | **100%** |
| A1 Ingestion | page 元数据覆盖率 | **100%** |
| A1 Ingestion | 平均 tokens/chunk（chunk_size=256）| **134.0** |
| A1 Ingestion | P95 tokens/chunk | **256**（上限触发符合预期）|
| A2 API | 单线程 P50 / P95 延迟 | **2.1ms / 2.5ms** |
| A2 API | 并发 8 线程 P50 / P95 延迟 | **13.7ms / 18.8ms** |
| A2 API | 吞吐量（并发 8）| **560 req/s** |
| A2 API | 错误率 | **0%** |
| A3 Citation | 检索 Recall@5（30 题 QA 集）| **90%** |
| A3 Citation | 支持段 vs 干扰段 NLI Δ | **+0.096**（有效区分）|
| A3 Citation | Page grounding accuracy | **1.000**（PDF 段落全有 page 元数据）|

报告全文：`report/phase2_eval.json`

---

## 5. 已知 Limitations / 下一步（Phase 3+）

| 项目 | 说明 |
|------|------|
| `/ingest` API | 当前需离线建 index，API 层 async ingest job 未实现 |
| OCR 支持 | `PdfParser` 只处理 native-text PDF，扫描件需 surya-ocr/marker |
| 真实 HHEM 验证 | 评测使用 overlap-stub；`--use-real-hhem` 需约 500MB 下载 |
| SPLADE / setwise rerank | 计划中的 B1/B2，未实现（Phase 2 时间预算内未到达） |
| PDF Q&A 标注数据集 | phase2 计划要求 ≥100 句子-passage 对的人工标注，未建立 |
| test_query_factory 预存 bug | 两个测试需补充 `LLM_BASE_URL` mock，与 Phase 2 无关 |

---

## 6. 面试叙事摘要

**技术亮点**：
1. **Document schema 无停机迁移**：`frozen=True` dataclass 扩展 + 条件序列化 + `hash=False` 修复，全量 Phase 1 数据零改动加载
2. **LazyDocstore 并发 bug 修复**：从共享 file handle 改为 per-request open，支持 FastAPI 多线程并发检索
3. **NLI attribution 与 LLM judge 并行**：post-hoc NLI 路径不修改任何现有 pipeline，通过注入式 `HHEMScorer` 实现，可直接用于 Phase 1 历史数据的回溯评分
4. **API 性能**：单线程 P95 < 3ms，并发 8 线程吞吐 560 req/s，零错误率

**转型价值**：从"离线 benchmark 脚本"到"可被 agent runtime 调用的 HTTP 知识服务"，为 `llm-coding-agent-system` 提供检索后端接口。
