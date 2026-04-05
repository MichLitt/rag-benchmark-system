"""Tests for the EvalOps integration layer (C1–C2).

All tests are fully offline — no network calls are made.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from src.evalops.adapter import build_eval_run_report
from src.evalops.client import EvalOpsClient
from src.evalops.schema import EvalRunReport
from src.types import RunExampleResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    query_id: str = "q1",
    *,
    retrieval_latency_ms: float = 10.0,
    rerank_latency_ms: float = 5.0,
    generation_latency_ms: float = 20.0,
    generation_cost_usd: float | None = None,
    retrieval_failure_bucket: str = "",
    failure_stage: str = "",
    failure_detail: str = "",
    run_id: str = "",
) -> RunExampleResult:
    return RunExampleResult(
        query_id=query_id,
        predicted_answer="Paris",
        gold_answers=["Paris"],
        retrieved_doc_ids=["d1"],
        retrieved_titles=["France"],
        unique_retrieved_titles=1,
        retrieval_latency_ms=retrieval_latency_ms,
        rerank_latency_ms=rerank_latency_ms,
        generation_latency_ms=generation_latency_ms,
        approx_input_tokens=100,
        approx_output_tokens=10,
        is_em=True,
        f1=1.0,
        recall_at_k=1.0,
        raw_candidate_count=20,
        dedup_candidate_count=20,
        duplicate_candidates_removed=0,
        generation_cost_usd=generation_cost_usd,
        retrieval_failure_bucket=retrieval_failure_bucket,
        failure_stage=failure_stage,
        failure_detail=failure_detail,
        run_id=run_id,
    )


_SAMPLE_METRICS = {
    "EM": 0.75,
    "F1": 0.80,
    "Recall@k": 0.90,
    "AvgRetrievalLatencyMs": 12.0,
    "AvgRerankLatencyMs": 6.0,
    "AvgGenerationLatencyMs": 25.0,
    "AvgQueryExpansionLatencyMs": 0.0,
    "NumQueries": 2,
    "Dataset": "hotpotqa",
    "Retriever": "dense",
    "GeneratorModel": "gpt-4o",
}


# ---------------------------------------------------------------------------
# EvalRunReport — schema defaults
# ---------------------------------------------------------------------------

def test_schema_version_default():
    report = EvalRunReport()
    assert report.schema_version == "rag/v1"


def test_schema_version_is_string():
    assert isinstance(EvalRunReport().schema_version, str)


def test_schema_defaults_are_zero_or_empty():
    report = EvalRunReport()
    assert report.run_id == ""
    assert report.em == 0.0
    assert report.f1 == 0.0
    assert report.recall_at_k == 0.0
    assert report.avg_faithfulness is None
    assert report.hallucination_rate is None
    assert report.total_generation_cost_usd is None
    assert report.retrieval_profile == []


def test_schema_is_dataclass():
    assert dataclasses.is_dataclass(EvalRunReport)


def test_schema_serialisable_with_asdict():
    report = EvalRunReport(run_id="r1", em=0.5)
    d = dataclasses.asdict(report)
    assert d["schema_version"] == "rag/v1"
    assert d["run_id"] == "r1"
    assert d["em"] == 0.5


def test_schema_asdict_json_roundtrip():
    report = EvalRunReport(run_id="r42", retrieval_profile=[{"query_id": "q1"}])
    payload = json.dumps(dataclasses.asdict(report))
    recovered = json.loads(payload)
    assert recovered["schema_version"] == "rag/v1"
    assert recovered["retrieval_profile"][0]["query_id"] == "q1"


# ---------------------------------------------------------------------------
# RunExampleResult — C2 traceability fields
# ---------------------------------------------------------------------------

def test_run_example_result_has_failure_stage():
    r = _make_result(failure_stage="retrieval")
    assert r.failure_stage == "retrieval"


def test_run_example_result_has_failure_detail():
    r = _make_result(failure_detail="index not found")
    assert r.failure_detail == "index not found"


def test_run_example_result_has_run_id():
    r = _make_result(run_id="run_20240101_120000")
    assert r.run_id == "run_20240101_120000"


def test_run_example_result_c2_fields_default_to_empty_string():
    r = _make_result()
    assert r.failure_stage == ""
    assert r.failure_detail == ""
    assert r.run_id == ""


# ---------------------------------------------------------------------------
# build_eval_run_report — adapter
# ---------------------------------------------------------------------------

def test_adapter_schema_version():
    results = [_make_result()]
    report = build_eval_run_report("run1", _SAMPLE_METRICS, results)
    assert report.schema_version == "rag/v1"


def test_adapter_copies_run_id():
    results = [_make_result()]
    report = build_eval_run_report("run_abc", _SAMPLE_METRICS, results)
    assert report.run_id == "run_abc"


def test_adapter_copies_metrics():
    results = [_make_result()]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    assert report.em == pytest.approx(0.75)
    assert report.f1 == pytest.approx(0.80)
    assert report.recall_at_k == pytest.approx(0.90)


def test_adapter_copies_latency():
    results = [_make_result()]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    assert report.avg_retrieval_latency_ms == pytest.approx(12.0)
    assert report.avg_rerank_latency_ms == pytest.approx(6.0)
    assert report.avg_generation_latency_ms == pytest.approx(25.0)


def test_adapter_dataset_from_metrics():
    results = [_make_result()]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    assert report.dataset == "hotpotqa"


def test_adapter_dataset_override():
    results = [_make_result()]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results, dataset="nq")
    assert report.dataset == "nq"


def test_adapter_retriever_from_metrics():
    results = [_make_result()]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    assert report.retriever_mode == "dense"


def test_adapter_num_queries_from_metrics():
    results = [_make_result()]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    assert report.num_queries == 2


def test_adapter_num_queries_falls_back_to_results_length():
    results = [_make_result("q1"), _make_result("q2"), _make_result("q3")]
    metrics_no_nq = {k: v for k, v in _SAMPLE_METRICS.items() if k != "NumQueries"}
    report = build_eval_run_report("r", metrics_no_nq, results)
    assert report.num_queries == 3


def test_adapter_faithfulness_is_none_by_default():
    results = [_make_result()]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    assert report.avg_faithfulness is None
    assert report.hallucination_rate is None


def test_adapter_cost_aggregation():
    results = [
        _make_result("q1", generation_cost_usd=0.01),
        _make_result("q2", generation_cost_usd=0.03),
    ]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    assert report.total_generation_cost_usd == pytest.approx(0.04)
    assert report.avg_generation_cost_usd == pytest.approx(0.02)


def test_adapter_cost_none_when_no_costs():
    results = [_make_result()]  # generation_cost_usd=None
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    assert report.total_generation_cost_usd is None
    assert report.avg_generation_cost_usd is None


def test_adapter_cost_skips_none_values():
    """Only non-None costs are aggregated."""
    results = [
        _make_result("q1", generation_cost_usd=0.05),
        _make_result("q2", generation_cost_usd=None),
    ]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    assert report.total_generation_cost_usd == pytest.approx(0.05)
    assert report.avg_generation_cost_usd == pytest.approx(0.05)


def test_adapter_retrieval_profile_length():
    results = [_make_result("q1"), _make_result("q2"), _make_result("q3")]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    assert len(report.retrieval_profile) == 3


def test_adapter_retrieval_profile_contains_query_id():
    results = [_make_result("qX")]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    assert report.retrieval_profile[0]["query_id"] == "qX"


def test_adapter_retrieval_profile_contains_latencies():
    results = [_make_result(retrieval_latency_ms=42.0, rerank_latency_ms=7.0, generation_latency_ms=99.0)]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    p = report.retrieval_profile[0]
    assert p["retrieval_latency_ms"] == pytest.approx(42.0)
    assert p["rerank_latency_ms"] == pytest.approx(7.0)
    assert p["generation_latency_ms"] == pytest.approx(99.0)


def test_adapter_retrieval_profile_contains_failure_fields():
    results = [_make_result(failure_stage="generation", failure_detail="timeout")]
    report = build_eval_run_report("r", _SAMPLE_METRICS, results)
    p = report.retrieval_profile[0]
    assert p["failure_stage"] == "generation"
    assert p["failure_detail"] == "timeout"


def test_adapter_empty_results():
    report = build_eval_run_report("r", _SAMPLE_METRICS, [])
    assert report.retrieval_profile == []
    assert report.total_generation_cost_usd is None


# ---------------------------------------------------------------------------
# EvalOpsClient — silent-fail behaviour
# ---------------------------------------------------------------------------

def test_client_submit_no_endpoint_is_noop():
    """Client with no endpoint must not raise."""
    client = EvalOpsClient()
    report = EvalRunReport(run_id="test")
    client.submit(report)  # should not raise


def test_client_from_env_returns_instance(monkeypatch):
    monkeypatch.setenv("EVALOPS_ENDPOINT", "http://fake.example/api")
    monkeypatch.setenv("EVALOPS_API_KEY", "tok123")
    client = EvalOpsClient.from_env()
    assert isinstance(client, EvalOpsClient)
    assert client._endpoint == "http://fake.example/api"
    assert client._api_key == "tok123"


def test_client_from_env_missing_vars(monkeypatch):
    monkeypatch.delenv("EVALOPS_ENDPOINT", raising=False)
    monkeypatch.delenv("EVALOPS_API_KEY", raising=False)
    client = EvalOpsClient.from_env()
    assert client._endpoint == ""
    assert client._api_key == ""


def test_client_submit_swallows_network_error():
    """_do_submit raises; submit() must still not propagate."""
    class BrokenClient(EvalOpsClient):
        def _do_submit(self, report):
            raise ConnectionRefusedError("no server")

    client = BrokenClient(endpoint="http://nowhere")
    client.submit(EvalRunReport(run_id="r"))  # must not raise


def test_client_submit_swallows_arbitrary_exception():
    class WeirdClient(EvalOpsClient):
        def _do_submit(self, report):
            raise ValueError("unexpected error")

    client = WeirdClient(endpoint="http://x")
    client.submit(EvalRunReport())  # must not raise


def test_client_do_submit_noop_with_empty_endpoint():
    """_do_submit with empty endpoint must not attempt any I/O."""
    client = EvalOpsClient(endpoint="")
    # This would raise if it tried urllib.request — it should just return
    client._do_submit(EvalRunReport(run_id="safe"))
