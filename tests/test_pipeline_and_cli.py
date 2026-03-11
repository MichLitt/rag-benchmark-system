import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_naive_rag_baseline import _make_progress_callback, parse_args
from src.generation.base import GenerationResult
from src.io_utils import save_run_results
from src.pipeline import run_naive_rag
from src.types import Document, QuerySample


class _FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        self.queries.append(query)
        return [
            Document(doc_id="d1", title="France", text="Paris"),
            Document(doc_id="d2", title="France", text="Paris duplicate"),
            Document(doc_id="d3", title="Europe", text="France is in Europe"),
        ][:top_k]


class _FakeHotpotRetriever:
    def retrieve(self, query: str, top_k: int) -> list[Document]:
        docs = [
            Document(doc_id="d1", title="Alpha", text="alpha chunk 1"),
            Document(doc_id="d2", title="Alpha", text="alpha chunk 2"),
            Document(doc_id="d3", title="Beta", text="beta chunk 1"),
            Document(doc_id="d4", title="Gamma", text="gamma chunk 1"),
        ]
        return docs[:top_k]


class _HotpotMetricsRetriever:
    def retrieve(self, query: str, top_k: int) -> list[Document]:
        mapping = {
            "q1": [
                Document(doc_id="d1", title="Alpha", text="alpha chunk"),
                Document(doc_id="d2", title="Beta", text="beta chunk"),
                Document(doc_id="d3", title="Gamma", text="gamma chunk"),
            ],
            "q2": [
                Document(doc_id="d4", title="Alpha", text="alpha chunk 1"),
                Document(doc_id="d5", title="Alpha", text="alpha chunk 2"),
                Document(doc_id="d6", title="Beta", text="beta chunk"),
            ],
        }
        return mapping[query][:top_k]


class _FakeMultiQueryRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        self.queries.append(query)
        mapping = {
            "What connects Tim Burton to the film Ed Wood?": [
                Document(doc_id="d0", title="Ed Wood", text="Ed Wood is a film by Tim Burton."),
                Document(doc_id="d1", title="Tim Burton", text="Tim Burton directed Ed Wood."),
            ],
            "Tim Burton Ed Wood film cast": [
                Document(doc_id="d1", title="Tim Burton", text="Tim Burton directed Ed Wood."),
                Document(doc_id="d2", title="Ed Wood", text="Johnny Depp starred in Ed Wood."),
            ],
            "Ed Wood film lead actor": [
                Document(doc_id="d2", title="Ed Wood", text="Johnny Depp starred in Ed Wood."),
                Document(doc_id="d3", title="Johnny Depp", text="Johnny Depp is an actor."),
            ],
        }
        return mapping[query][:top_k]


class _FakeBatchMultiQueryRetriever(_FakeMultiQueryRetriever):
    def __init__(self) -> None:
        super().__init__()
        self.retrieve_many_calls: list[list[str]] = []

    def retrieve_many(self, queries: list[str], top_k: int) -> list[list[Document]]:
        self.retrieve_many_calls.append(list(queries))
        return [self.retrieve(query, top_k) for query in queries]


class _FakeGenerator:
    def generate(self, question: str, contexts: list[Document]) -> GenerationResult:
        return GenerationResult(
            text="Paris",
            input_tokens=20,
            output_tokens=7,
            reasoning_tokens=5,
            provider="fake",
            model="fake-model",
        )


class _ReverseReranker:
    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        return list(reversed(docs))[:top_k]


class _FailingGenerator:
    def generate(self, question: str, contexts: list[Document]) -> GenerationResult:
        raise RuntimeError("budget exhausted")


class _FakeExpander:
    def __init__(self, expanded: str) -> None:
        self.expanded = expanded
        self.questions: list[str] = []

    def expand(self, question: str) -> str:
        self.questions.append(question)
        return self.expanded


class _FakeMultiQueryExpander:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries
        self.questions: list[str] = []

    def expand(self, question: str) -> str:
        self.questions.append(question)
        return " || ".join(self.queries)

    def expand_queries(self, question: str) -> list[str]:
        self.questions.append(question)
        return list(self.queries)


class _FailingExpander:
    def expand(self, question: str) -> str:
        raise RuntimeError("hyde unavailable")


class PipelineAndCliTests(unittest.TestCase):
    def test_parse_args_accepts_query_expansion_cli_flags(self):
        with patch.object(
            sys,
            "argv",
            [
                "run_naive_rag_baseline.py",
                "--generator-max-completion-tokens",
                "512",
                "--generator-reasoning-split",
                "true",
                "--dedup-mode",
                "title",
                "--dedup-before-rerank",
                "--reranker-retriever-rank-weight",
                "0.25",
                "--reranker-rank-fusion-k",
                "30",
                "--query-expansion-mode",
                "auto",
                "--query-expansion-model",
                "MiniMax-M2.5",
                "--query-expansion-datasets",
                "hotpotqa,nq",
                "--query-expansion-max-completion-tokens",
                "96",
                "--qa-path",
                "data/filtered/custom_hotpot_subset.jsonl",
                "--progress-every",
                "10",
                "--progress-min-seconds",
                "5",
                "--continue-on-generation-error",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.generator_max_completion_tokens, 512)
        self.assertEqual(args.generator_reasoning_split, "true")
        self.assertEqual(args.dedup_mode, "title")
        self.assertTrue(args.dedup_before_rerank)
        self.assertEqual(args.reranker_retriever_rank_weight, 0.25)
        self.assertEqual(args.reranker_rank_fusion_k, 30)
        self.assertEqual(args.query_expansion_mode, "auto")
        self.assertEqual(args.query_expansion_model, "MiniMax-M2.5")
        self.assertEqual(args.query_expansion_datasets, "hotpotqa,nq")
        self.assertEqual(args.query_expansion_max_completion_tokens, 96)
        self.assertEqual(args.qa_path, Path("data/filtered/custom_hotpot_subset.jsonl"))
        self.assertEqual(args.progress_every, 10)
        self.assertEqual(args.progress_min_seconds, 5)
        self.assertTrue(args.continue_on_generation_error)

    def test_pipeline_records_reasoning_token_metrics(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="What is the capital of France?",
                answers=["Paris"],
                gold_doc_id="d1",
            )
        ]

        results, metrics = run_naive_rag(
            retriever=_FakeRetriever(),
            eval_set=samples,
            top_k=1,
            generator=_FakeGenerator(),
        )

        self.assertEqual(results[0].actual_reasoning_tokens, 5)
        self.assertEqual(metrics["AvgReasoningTokensActual"], 5)

    def test_pipeline_deduplicates_by_title(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="What is the capital of France?",
                answers=["Paris"],
                gold_doc_id="d1",
            )
        ]

        results, metrics = run_naive_rag(
            retriever=_FakeRetriever(),
            eval_set=samples,
            top_k=2,
            retrieve_top_k=3,
            generator=_FakeGenerator(),
            dedup_mode="title",
            dedup_before_rerank=False,
        )

        self.assertEqual(results[0].retrieved_doc_ids, ["d1", "d3"])
        self.assertEqual(results[0].unique_retrieved_titles, 2)
        self.assertEqual(results[0].duplicate_candidates_removed, 1)
        self.assertEqual(metrics["AvgUniqueTitles@k"], 2)

    def test_pipeline_reports_hotpot_all_gold_metrics_and_failure_buckets(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="q1",
                answers=["alpha"],
                gold_titles=["Alpha", "Beta"],
            ),
            QuerySample(
                query_id="q2",
                question="q2",
                answers=["alpha"],
                gold_titles=["Alpha", "Beta"],
            ),
        ]

        _, metrics = run_naive_rag(
            retriever=_HotpotMetricsRetriever(),
            eval_set=samples,
            top_k=2,
            retrieve_top_k=3,
            generator=_FakeGenerator(),
        )

        self.assertEqual(metrics["RecallAnyGoldTitle@k"], 1.0)
        self.assertEqual(metrics["RecallAllGold@k_title"], 0.5)
        self.assertEqual(metrics["RecallAllGold@raw_title"], 1.0)
        self.assertEqual(
            metrics["FailureBucketCounts"],
            {
                "both_gold_in_final": 1,
                "both_gold_after_dedup_but_lost_after_rerank": 1,
            },
        )

    def test_pipeline_keeps_reranker_order_after_dedup_before_rerank(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="What is the capital of France?",
                answers=["Paris"],
                gold_doc_id="d3",
            )
        ]

        results, _ = run_naive_rag(
            retriever=_FakeRetriever(),
            eval_set=samples,
            top_k=2,
            retrieve_top_k=3,
            generator=_FakeGenerator(),
            reranker=_ReverseReranker(),
            dedup_mode="title",
            dedup_before_rerank=True,
        )

        self.assertEqual(results[0].retrieved_doc_ids, ["d3", "d1"])

    def test_pipeline_uses_expanded_query_for_retrieval(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="What is the capital of France?",
                answers=["Paris"],
                gold_doc_id="d1",
            )
        ]
        retriever = _FakeRetriever()
        expander = _FakeExpander("France capital city Paris Europe")

        results, metrics = run_naive_rag(
            retriever=retriever,
            eval_set=samples,
            top_k=1,
            generator=_FakeGenerator(),
            query_expander=expander,
            query_expansion_mode="hyde",
        )

        self.assertEqual(retriever.queries, ["France capital city Paris Europe"])
        self.assertEqual(results[0].expanded_query, "France capital city Paris Europe")
        self.assertEqual(results[0].query_expansion_mode, "hyde")
        self.assertEqual(metrics["QueryExpansionMode"], "hyde")
        self.assertEqual(metrics["NumQueryExpansionErrors"], 0)

    def test_pipeline_calls_progress_callback(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="What is the capital of France?",
                answers=["Paris"],
                gold_doc_id="d1",
            )
        ]
        calls: list[tuple[int, int]] = []

        def _progress(results, processed_queries, total_queries):
            calls.append((processed_queries, total_queries))

        run_naive_rag(
            retriever=_FakeRetriever(),
            eval_set=samples,
            top_k=1,
            generator=_FakeGenerator(),
            progress_callback=_progress,
        )

        self.assertEqual(calls, [(1, 1)])

    def test_pipeline_fuses_multi_query_retrieval(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="What connects Tim Burton to the film Ed Wood?",
                answers=["Johnny Depp"],
                gold_doc_id="d2",
            )
        ]
        retriever = _FakeMultiQueryRetriever()
        expander = _FakeMultiQueryExpander(
            [
                "What connects Tim Burton to the film Ed Wood?",
                "Tim Burton Ed Wood film cast",
                "Ed Wood film lead actor",
            ]
        )

        results, metrics = run_naive_rag(
            retriever=retriever,
            eval_set=samples,
            top_k=2,
            generator=_FakeGenerator(),
            query_expander=expander,
            query_expansion_mode="hotpot_decompose",
        )

        self.assertEqual(
            retriever.queries,
            [
                "What connects Tim Burton to the film Ed Wood?",
                "Tim Burton Ed Wood film cast",
                "Ed Wood film lead actor",
            ],
        )
        self.assertEqual(
            results[0].expanded_queries,
            [
                "What connects Tim Burton to the film Ed Wood?",
                "Tim Burton Ed Wood film cast",
                "Ed Wood film lead actor",
            ],
        )
        self.assertEqual(results[0].retrieved_doc_ids, ["d1", "d2"])
        self.assertEqual(metrics["AvgExpandedQueriesPerSample"], 3.0)

    def test_pipeline_uses_retrieve_many_when_available_for_multi_query_fusion(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="What connects Tim Burton to the film Ed Wood?",
                answers=["Johnny Depp"],
                gold_doc_id="d2",
            )
        ]
        retriever = _FakeBatchMultiQueryRetriever()
        expander = _FakeMultiQueryExpander(
            [
                "What connects Tim Burton to the film Ed Wood?",
                "Tim Burton Ed Wood film cast",
                "Ed Wood film lead actor",
            ]
        )

        results, _ = run_naive_rag(
            retriever=retriever,
            eval_set=samples,
            top_k=2,
            generator=_FakeGenerator(),
            query_expander=expander,
            query_expansion_mode="hotpot_decompose",
        )

        self.assertEqual(
            retriever.retrieve_many_calls,
            [[
                "What connects Tim Burton to the film Ed Wood?",
                "Tim Burton Ed Wood film cast",
                "Ed Wood film lead actor",
            ]],
        )
        self.assertEqual(results[0].retrieved_doc_ids, ["d1", "d2"])

    def test_pipeline_falls_back_to_original_query_on_query_expansion_error(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="What is the capital of France?",
                answers=["Paris"],
                gold_doc_id="d1",
            )
        ]
        retriever = _FakeRetriever()

        results, metrics = run_naive_rag(
            retriever=retriever,
            eval_set=samples,
            top_k=1,
            generator=_FakeGenerator(),
            query_expander=_FailingExpander(),
            query_expansion_mode="hyde",
        )

        self.assertEqual(retriever.queries, ["What is the capital of France?"])
        self.assertEqual(results[0].expanded_query, "What is the capital of France?")
        self.assertIn("hyde unavailable", results[0].query_expansion_error)
        self.assertEqual(metrics["NumQueryExpansionErrors"], 1)
        self.assertEqual(metrics["QueryExpansionErrorRate"], 1.0)

    def test_pipeline_can_record_query_expansion_and_generation_errors_together(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="What is the capital of France?",
                answers=["Paris"],
                gold_doc_id="d1",
            )
        ]

        results, metrics = run_naive_rag(
            retriever=_FakeRetriever(),
            eval_set=samples,
            top_k=1,
            generator=_FailingGenerator(),
            query_expander=_FailingExpander(),
            query_expansion_mode="hyde",
            continue_on_generation_error=True,
        )

        self.assertIn("hyde unavailable", results[0].query_expansion_error)
        self.assertIn("budget exhausted", results[0].generation_error)
        self.assertEqual(metrics["NumQueryExpansionErrors"], 1)
        self.assertEqual(metrics["NumGenerationErrors"], 1)

    def test_save_run_results_writes_query_expansion_fields(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="What is the capital of France?",
                answers=["Paris"],
                gold_doc_id="d1",
            )
        ]
        retriever = _FakeRetriever()
        expander = _FakeExpander("France capital city Paris Europe")
        results, _ = run_naive_rag(
            retriever=retriever,
            eval_set=samples,
            top_k=1,
            generator=_FakeGenerator(),
            query_expander=expander,
            query_expansion_mode="hyde",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "predictions.json"
            save_run_results(output_path, results)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload[0]["expanded_query"], "France capital city Paris Europe")
        self.assertEqual(payload[0]["expanded_queries"], ["France capital city Paris Europe"])
        self.assertEqual(payload[0]["query_expansion_mode"], "hyde")
        self.assertIn("query_expansion_latency_ms", payload[0])
        self.assertEqual(payload[0]["query_expansion_error"], "")
        self.assertIn("query_expansion_failure_reason", payload[0])
        self.assertIn("query_expansion_cache_key", payload[0])
        self.assertIn("query_expansion_used_fallback", payload[0])
        self.assertIn("retrieval_failure_bucket", payload[0])

    def test_pipeline_can_continue_on_generation_error(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="What is the capital of France?",
                answers=["Paris"],
                gold_doc_id="d1",
            )
        ]

        results, metrics = run_naive_rag(
            retriever=_FakeRetriever(),
            eval_set=samples,
            top_k=1,
            generator=_FailingGenerator(),
            continue_on_generation_error=True,
        )

        self.assertEqual(results[0].predicted_answer, "")
        self.assertIn("budget exhausted", results[0].generation_error)
        self.assertEqual(metrics["NumGenerationErrors"], 1)
        self.assertEqual(metrics["GenerationErrorRate"], 1.0)

    def test_progress_callback_writes_heartbeat_and_partial_predictions(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="What is the capital of France?",
                answers=["Paris"],
                gold_doc_id="d1",
            )
        ]
        results, _ = run_naive_rag(
            retriever=_FakeRetriever(),
            eval_set=samples,
            top_k=1,
            generator=_FakeGenerator(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "hotpotqa"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            qa_path = Path(tmp_dir) / "qa.jsonl"
            qa_path.write_text('{"id":"q1"}\n', encoding="utf-8")

            progress_callback, write_completed, _ = _make_progress_callback(
                dataset_dir=dataset_dir,
                dataset_name="hotpotqa",
                qa_path=qa_path,
                retriever_mode="dense_sharded",
                query_expansion_mode="off",
                use_reranker=False,
                total_queries=1,
                progress_every=1,
                progress_min_seconds=9999,
            )
            progress_callback(results, 1, 1)

            progress_payload = json.loads((dataset_dir / "progress.json").read_text(encoding="utf-8"))
            partial_rows = [
                json.loads(line)
                for line in (dataset_dir / "predictions.partial.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(progress_payload["status"], "running")
            self.assertEqual(progress_payload["processed_queries"], 1)
            self.assertEqual(progress_payload["metrics_snapshot"]["Recall@k"], 1.0)
            self.assertIn("QueryExpansionFallbackRate", progress_payload["metrics_snapshot"])
            self.assertEqual(partial_rows[0]["query_id"], "q1")

            write_completed(results)
            completed_payload = json.loads((dataset_dir / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(completed_payload["status"], "completed")

    def test_pipeline_title_first_rerank_packs_diverse_titles_before_extra_chunks(self):
        samples = [
            QuerySample(
                query_id="q1",
                question="Alpha Beta relation?",
                answers=["answer"],
                gold_titles=["Alpha", "Beta"],
            )
        ]

        results, metrics = run_naive_rag(
            retriever=_FakeHotpotRetriever(),
            eval_set=samples,
            top_k=3,
            retrieve_top_k=4,
            generator=_FakeGenerator(),
            reranker=_ReverseReranker(),
            dedup_mode="title",
            dedup_before_rerank=True,
            title_first_rerank=True,
            title_pool_k=3,
            max_chunks_per_title=2,
            min_unique_titles=2,
        )

        self.assertEqual(results[0].retrieved_doc_ids, ["d4", "d3", "d1"])
        self.assertEqual(results[0].gold_titles_in_raw_candidates, ["Alpha", "Beta"])
        self.assertEqual(results[0].gold_titles_after_dedup, ["Alpha", "Beta"])
        self.assertEqual(results[0].gold_titles_in_final_top_k, ["Alpha", "Beta"])
        self.assertEqual(results[0].retrieval_failure_bucket, "both_gold_in_final")
        self.assertTrue(metrics["TitleFirstRerank"])


if __name__ == "__main__":
    unittest.main()
