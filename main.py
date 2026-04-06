from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from src.config import load_yaml_config
from src.evalops.adapter import build_eval_run_report
from src.evalops.client import EvalOpsClient
from src.generation import build_generator
from src.io_utils import ensure_dir, save_json, save_run_results
from src.pipeline import run_naive_rag
from src.retrieval.factory import build_retriever
from src.types import Document, QuerySample


def build_toy_data() -> tuple[list[Document], list[QuerySample]]:
    corpus = [
        Document(doc_id="d1", text="Paris is the capital of France."),
        Document(doc_id="d2", text="Tokyo is the capital of Japan."),
        Document(doc_id="d3", text="The Pacific Ocean is the largest ocean on Earth."),
    ]
    eval_set = [
        QuerySample(
            query_id="q1",
            question="What is the capital of France?",
            answers=["Paris"],
            gold_doc_id="d1",
        ),
        QuerySample(
            query_id="q2",
            question="Which ocean is the largest on Earth?",
            answers=["Pacific Ocean"],
            gold_doc_id="d3",
        ),
    ]
    return corpus, eval_set


def main() -> None:
    cfg = load_yaml_config("config/default.yaml")
    top_k = int(cfg["retrieval"]["top_k"])

    corpus, eval_set = build_toy_data()
    retriever = build_retriever(cfg, corpus=corpus)
    generator = build_generator(cfg)
    results, metrics = run_naive_rag(retriever, eval_set, top_k=top_k, generator=generator)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(Path("experiments") / "runs" / run_id)
    save_json(out_dir / "metrics.json", metrics)
    # Stamp run_id for per-example traceability before serializing
    for r in results:
        r.run_id = run_id
    save_run_results(out_dir / "predictions.json", results)

    # EvalOps: submit run report (fails silently if endpoint not configured)
    report = build_eval_run_report(run_id, metrics, results)
    EvalOpsClient.from_env().submit(report)

    print("Run finished")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
