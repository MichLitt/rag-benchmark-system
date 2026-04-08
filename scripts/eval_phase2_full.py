#!/usr/bin/env python3
"""Phase 2 全量评测脚本

量化 Phase 2 四个组件的实际效果：
  A0  Document schema — 向后兼容性与字段完整性
  A1  PDF ingestion   — chunk 质量（token 分布、页面覆盖、重叠率）
  A2  FastAPI service — 并发检索延迟（P50/P95/P99）与吞吐量
  A3  NLI citation    — attribution rate / supporting hit / page grounding

评测使用内置合成语料库（无需 FAISS、大模型或外部 API），
通过 --use-real-hhem 可接入 Vectara HHEM（需下载 ~500MB 模型）。

Usage:
    uv run python scripts/eval_phase2_full.py
    uv run python scripts/eval_phase2_full.py --use-real-hhem
    uv run python scripts/eval_phase2_full.py --output report/phase2_eval.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tiktoken

from src.evaluation.citation import CitationEvaluator, CitationResult
from src.evaluation.hhem_scorer import HHEMResult, HHEMScorer
from src.ingestion.chunker import TokenAwareChunker
from src.ingestion.pdf_parser import PdfParser
from src.retrieval.docstore import build_docstore_offsets, load_docstore, save_docstore, LazyDocstore
from src.retrieval.keyword import KeywordRetriever
from src.types import Document, ScoredDocument

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic QA corpus — 30 factual Q&A pairs with source documents
# Each entry: question / gold_answer / supporting_text / distractor_text
# ─────────────────────────────────────────────────────────────────────────────
SYNTHETIC_QA: list[dict] = [
    {"q": "What is the capital of France?",
     "a": "Paris",
     "supp": "Paris is the capital and most populous city of France. It is located in northern France on the Seine River.",
     "dist": "Lyon is a major city in southeastern France, known for its gastronomy and silk industry."},
    {"q": "What gas do plants absorb during photosynthesis?",
     "a": "carbon dioxide",
     "supp": "During photosynthesis, plants absorb carbon dioxide from the air and convert it into glucose using sunlight and water.",
     "dist": "Oxygen is a gas produced by plants as a byproduct of photosynthesis through the light-dependent reactions."},
    {"q": "Who wrote the play Hamlet?",
     "a": "Shakespeare",
     "supp": "Hamlet is a tragedy written by William Shakespeare around 1600–1601. It is one of his most famous works.",
     "dist": "Macbeth is another famous tragedy by Shakespeare, depicting the rise and fall of a Scottish king."},
    {"q": "What is the speed of light in a vacuum?",
     "a": "299792458 meters per second",
     "supp": "The speed of light in a vacuum is exactly 299792458 meters per second, approximately 3×10^8 m/s.",
     "dist": "The speed of sound in air at sea level is approximately 343 meters per second at 20 degrees Celsius."},
    {"q": "What year did World War II end?",
     "a": "1945",
     "supp": "World War II ended in 1945. Germany surrendered on May 8, 1945 (V-E Day) and Japan on September 2, 1945.",
     "dist": "World War I ended in 1918 with the signing of the Armistice at Compiègne on November 11."},
    {"q": "What is the chemical formula for water?",
     "a": "H2O",
     "supp": "Water has the chemical formula H2O, meaning each molecule contains two hydrogen atoms bonded to one oxygen atom.",
     "dist": "Carbon dioxide has the chemical formula CO2, consisting of one carbon atom bonded to two oxygen atoms."},
    {"q": "What is the largest planet in the solar system?",
     "a": "Jupiter",
     "supp": "Jupiter is the largest planet in the solar system, with a mass more than twice that of all other planets combined.",
     "dist": "Saturn is the second largest planet, famous for its prominent ring system made of ice and rock."},
    {"q": "What is the powerhouse of the cell?",
     "a": "mitochondria",
     "supp": "The mitochondria are often called the powerhouse of the cell because they generate ATP through cellular respiration.",
     "dist": "The nucleus is the control center of the cell, containing the cell's DNA and directing cellular activities."},
    {"q": "In what city was the Eiffel Tower built?",
     "a": "Paris",
     "supp": "The Eiffel Tower was built in Paris, France, as the entrance arch for the 1889 World's Fair. It stands 330 meters tall.",
     "dist": "The Louvre Museum, also located in Paris, is the world's largest art museum and holds the Mona Lisa."},
    {"q": "What is the atomic number of carbon?",
     "a": "6",
     "supp": "Carbon has atomic number 6, meaning each carbon atom contains 6 protons in its nucleus.",
     "dist": "Oxygen has atomic number 8, with 8 protons in its nucleus and is essential for aerobic respiration."},
    {"q": "Who painted the Mona Lisa?",
     "a": "Leonardo da Vinci",
     "supp": "The Mona Lisa was painted by Leonardo da Vinci, likely between 1503 and 1519. It is housed in the Louvre Museum in Paris.",
     "dist": "Michelangelo painted the Sistine Chapel ceiling between 1508 and 1512, including the famous Creation of Adam."},
    {"q": "What is the boiling point of water at standard pressure?",
     "a": "100 degrees Celsius",
     "supp": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (1 atm).",
     "dist": "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit) at standard atmospheric pressure."},
    {"q": "What programming language was Python named after?",
     "a": "Monty Python",
     "supp": "Python was named after the British comedy series Monty Python's Flying Circus, not the snake. Creator Guido van Rossum was a fan.",
     "dist": "Java was named after Java coffee, which is a type of coffee from Java island in Indonesia."},
    {"q": "What is the longest river in the world?",
     "a": "Nile",
     "supp": "The Nile River, flowing through northeastern Africa, is generally considered the longest river in the world at about 6,650 km.",
     "dist": "The Amazon River in South America is the largest river by water discharge and contains the greatest biodiversity."},
    {"q": "How many bones are in the adult human body?",
     "a": "206",
     "supp": "The adult human body has 206 bones. Babies are born with around 270–300 bones, which fuse together as they grow.",
     "dist": "The human body contains over 600 muscles, which make up about 40% of total body weight in adults."},
    {"q": "What is the currency of Japan?",
     "a": "yen",
     "supp": "The Japanese yen is the official currency of Japan. It is the third most traded currency in the foreign exchange market.",
     "dist": "The South Korean won is the official currency of South Korea, introduced in 1945 after independence."},
    {"q": "What is the hardest natural substance on Earth?",
     "a": "diamond",
     "supp": "Diamond is the hardest natural substance on Earth, scoring 10 on the Mohs hardness scale. It is a form of carbon.",
     "dist": "Corundum, which includes rubies and sapphires, scores 9 on the Mohs scale and is used as an abrasive material."},
    {"q": "Which planet is closest to the Sun?",
     "a": "Mercury",
     "supp": "Mercury is the closest planet to the Sun and the smallest planet in the solar system. It has no atmosphere.",
     "dist": "Venus is the second planet from the Sun and the hottest planet, with surface temperatures reaching 465°C."},
    {"q": "What is the main function of red blood cells?",
     "a": "carry oxygen",
     "supp": "Red blood cells carry oxygen from the lungs to tissues throughout the body using hemoglobin, and transport carbon dioxide back.",
     "dist": "White blood cells are part of the immune system, defending the body against infections and foreign substances."},
    {"q": "What is the smallest unit of matter?",
     "a": "atom",
     "supp": "The atom is the smallest unit of ordinary matter that retains the chemical properties of an element.",
     "dist": "Molecules are groups of two or more atoms bonded together and represent the smallest unit of a chemical compound."},
    {"q": "In what year did humans first land on the Moon?",
     "a": "1969",
     "supp": "In 1969, NASA's Apollo 11 mission successfully landed humans on the Moon. Neil Armstrong was the first person to walk on the lunar surface.",
     "dist": "In 1957, the Soviet Union launched Sputnik 1, the first artificial satellite, marking the beginning of the Space Age."},
    {"q": "What is the chemical symbol for gold?",
     "a": "Au",
     "supp": "The chemical symbol for gold is Au, derived from the Latin word aurum. Gold has atomic number 79.",
     "dist": "The chemical symbol for silver is Ag, from the Latin argentum. Silver has atomic number 47."},
    {"q": "What ocean is the largest?",
     "a": "Pacific Ocean",
     "supp": "The Pacific Ocean is the largest and deepest ocean, covering more than 30% of Earth's surface and containing over 25,000 islands.",
     "dist": "The Atlantic Ocean is the second largest ocean and separates the Americas from Europe and Africa."},
    {"q": "What is the tallest mountain in the world?",
     "a": "Mount Everest",
     "supp": "Mount Everest, located in the Himalayas on the Nepal-Tibet border, is the tallest mountain at 8,848.86 meters above sea level.",
     "dist": "K2, located in Pakistan, is the second tallest mountain at 8,611 meters and is considered harder to climb than Everest."},
    {"q": "Who developed the theory of general relativity?",
     "a": "Albert Einstein",
     "supp": "Albert Einstein developed the theory of general relativity, published in 1915. It describes gravity as the curvature of spacetime.",
     "dist": "Isaac Newton formulated the laws of motion and universal gravitation in the 17th century, forming the foundation of classical mechanics."},
    {"q": "What is the most abundant gas in Earth's atmosphere?",
     "a": "nitrogen",
     "supp": "Nitrogen makes up about 78% of Earth's atmosphere by volume, making it the most abundant gas. Oxygen comprises about 21%.",
     "dist": "Carbon dioxide comprises only about 0.04% of Earth's atmosphere but plays a crucial role in the greenhouse effect."},
    {"q": "What country has the largest population?",
     "a": "India",
     "supp": "India surpassed China in 2023 to become the world's most populous country, with over 1.4 billion people.",
     "dist": "China has a population of approximately 1.4 billion people and held the title of most populous country for decades."},
    {"q": "What is the basic unit of heredity?",
     "a": "gene",
     "supp": "A gene is the basic unit of heredity. Genes are segments of DNA that carry instructions for producing proteins.",
     "dist": "A chromosome is a structure made of DNA and proteins that carries genes. Humans have 23 pairs of chromosomes."},
    {"q": "What is the freezing point of water in Fahrenheit?",
     "a": "32 degrees Fahrenheit",
     "supp": "Water freezes at 32 degrees Fahrenheit (0 degrees Celsius) at standard atmospheric pressure.",
     "dist": "Water boils at 212 degrees Fahrenheit (100 degrees Celsius) at standard atmospheric pressure."},
    {"q": "What is the square root of 144?",
     "a": "12",
     "supp": "The square root of 144 is 12, because 12 multiplied by 12 equals 144.",
     "dist": "The square root of 169 is 13, because 13 multiplied by 13 equals 169."},
]


# ─────────────────────────────────────────────────────────────────────────────
# Stub NLI scorer — word-overlap based (no model download required)
# ─────────────────────────────────────────────────────────────────────────────
class _OverlapScorer:
    """Stub scorer using token F1 overlap between passage and answer.

    Models the idea that a passage supporting an answer will share vocabulary.
    threshold=0.15 tuned so that the correct supporting passage scores >= threshold
    while most distractors score below.
    """
    def __init__(self, threshold: float = 0.15) -> None:
        self._threshold = threshold
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _token_overlap_f1(self, a: str, b: str) -> float:
        ta = set(self._enc.encode(a.lower()))
        tb = set(self._enc.encode(b.lower()))
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        prec = inter / len(ta)
        rec = inter / len(tb)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    def score(self, source: str, summary: str) -> HHEMResult:
        s = self._token_overlap_f1(source, summary)
        return HHEMResult(score=s, is_consistent=s >= self._threshold)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[HHEMResult]:
        return [self.score(src, summ) for src, summ in pairs]


# ─────────────────────────────────────────────────────────────────────────────
# Minimal PDF builder (no external library required)
# ─────────────────────────────────────────────────────────────────────────────
def _build_pdf_bytes(pages: list[str]) -> bytes:
    """Build a multi-page PDF from a list of text strings (one per page)."""
    def _escape(t: str) -> str:
        return t.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    header = b"%PDF-1.4\n"

    # Object IDs: 1=catalog, 2=pages, 3..N+2=page objects, N+3..2N+2=streams
    n = len(pages)
    page_ids = list(range(3, 3 + n))
    stream_ids = list(range(3 + n, 3 + 2 * n))
    kids_ref = b" ".join(f"{pid} 0 R".encode() for pid in page_ids)

    objects: dict[int, bytes] = {}

    # Catalog
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    # Pages collection
    objects[2] = b"<< /Type /Pages /Kids [" + kids_ref + b"] /Count " + str(n).encode() + b" >>"

    for i, (pid, sid) in enumerate(zip(page_ids, stream_ids)):
        # Content stream
        lines = _escape(pages[i])
        # Wrap at 80 chars to simulate multi-line text
        words = lines.split()
        wrapped: list[str] = []
        line_buf: list[str] = []
        for w in words:
            line_buf.append(w)
            if len(" ".join(line_buf)) > 70:
                wrapped.append(" ".join(line_buf[:-1]))
                line_buf = [w]
        if line_buf:
            wrapped.append(" ".join(line_buf))

        stream_lines = []
        for li, text_line in enumerate(wrapped):
            # Strip non-latin-1 characters to keep PDF content stream valid
            safe_line = text_line.encode("latin-1", errors="ignore").decode("latin-1")
            stream_lines.append(f"BT /F1 11 Tf 40 {720 - li*15} Td ({_escape(safe_line)}) Tj ET")
        stream_content = "\n".join(stream_lines).encode("latin-1")

        objects[sid] = (
            b"<< /Length " + str(len(stream_content)).encode() + b" >>\nstream\n"
            + stream_content + b"\nendstream"
        )
        # Page object
        font_dict = b"<< /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >>"
        objects[pid] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents " + str(sid).encode() + b" 0 R "
            b"/Resources << /Font " + font_dict + b" >> >>"
        )

    # Assemble body and cross-reference table
    body = b""
    xref_offsets: list[int] = []
    pos = len(header)
    max_id = max(objects)

    for oid in range(1, max_id + 1):
        xref_offsets.append(pos)
        obj_bytes = f"{oid} 0 obj\n".encode() + objects[oid] + b"\nendobj\n"
        body += obj_bytes
        pos += len(obj_bytes)

    xref_pos = len(header) + len(body)
    xref = b"xref\n0 " + str(max_id + 1).encode() + b"\n"
    xref += b"0000000000 65535 f \n"
    for off in xref_offsets:
        xref += f"{off:010d} 00000 n \n".encode()

    trailer = (
        b"trailer\n<< /Size " + str(max_id + 1).encode() + b" /Root 1 0 R >>\n"
        b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )

    return header + body + xref + trailer


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation sections
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ms(ms: float) -> str:
    return f"{ms:.1f}ms"


def eval_a0_schema(tmpdir: Path) -> dict:
    """A0: verify new-field round-trip and backward-compat at scale."""
    print("\n[A0] Document schema migration")

    n_new = 500
    n_old = 500
    new_docs = [
        Document(
            doc_id=f"new_{i}", text=f"content {i} " * 10, title=f"Title {i}",
            page_start=i + 1, page_end=i + 1,
            section="body" if i % 2 == 0 else "intro",
            source="corpus.pdf",
            extra_metadata={"chunk_index": i % 5},
        )
        for i in range(n_new)
    ]

    # Save new-format docstore
    new_path = tmpdir / "new_docs.jsonl"
    offsets_path = tmpdir / "new_docs.offsets"
    save_docstore(new_path, new_docs)
    build_docstore_offsets(new_path, offsets_path)

    # Reload and verify fields preserved
    loaded = load_docstore(new_path)
    fields_preserved = sum(
        1 for d in loaded
        if d.page_start is not None and d.section is not None and d.extra_metadata
    )

    # Write old-format JSONL (simulate pre-migration data)
    old_path = tmpdir / "old_docs.jsonl"
    with old_path.open("w", encoding="utf-8") as f:
        import json as _json
        for i in range(n_old):
            f.write(_json.dumps({"doc_id": f"old_{i}", "title": f"T{i}", "text": f"legacy text {i}"}) + "\n")
    old_loaded = load_docstore(old_path)
    compat_ok = sum(1 for d in old_loaded if d.page_start is None and d.extra_metadata == {})

    # LazyDocstore random-access test
    store = LazyDocstore(new_path, offsets_path)
    import random
    rng = random.Random(42)
    sample_idx = [rng.randint(0, n_new - 1) for _ in range(100)]
    lazy_ok = sum(1 for i in sample_idx if store.get(i).doc_id == f"new_{i}")

    result = {
        "new_docs_written": n_new,
        "new_fields_preserved_pct": fields_preserved / n_new * 100,
        "old_docs_backward_compat_pct": compat_ok / n_old * 100,
        "lazy_docstore_random_access_accuracy": lazy_ok / 100 * 100,
    }
    print(f"  new fields preserved:          {result['new_fields_preserved_pct']:.1f}%")
    print(f"  old-format backward compat:    {result['old_docs_backward_compat_pct']:.1f}%")
    print(f"  LazyDocstore random access ok: {result['lazy_docstore_random_access_accuracy']:.1f}%")
    return result


def eval_a1_ingestion(tmpdir: Path) -> dict:
    """A1: measure ingestion pipeline — pages, chunks, token distribution."""
    print("\n[A1] PDF ingestion pipeline")

    # Build synthetic PDFs of varying sizes
    pdf_specs = [
        ("short",  [SYNTHETIC_QA[i]["supp"] + " " + SYNTHETIC_QA[i]["dist"] for i in range(5)]),
        ("medium", [SYNTHETIC_QA[i]["supp"] * 4 for i in range(10)]),
        ("long",   [SYNTHETIC_QA[i]["supp"] * 12 for i in range(8)]),
    ]

    parser = PdfParser()
    chunker_256 = TokenAwareChunker(chunk_size=256, overlap=32)
    chunker_128 = TokenAwareChunker(chunk_size=128, overlap=16)
    enc = tiktoken.get_encoding("cl100k_base")

    total_pages = 0
    total_chunks_256 = 0
    total_chunks_128 = 0
    all_token_counts: list[int] = []
    pages_skipped = 0
    page_meta_coverage = 0  # chunks with page_start != None

    for name, page_texts in pdf_specs:
        pdf_bytes = _build_pdf_bytes(page_texts)
        pdf_path = tmpdir / f"{name}.pdf"
        pdf_path.write_bytes(pdf_bytes)

        pages = parser.parse(pdf_path)
        pages_with_text = len(pages)
        total_pages += pages_with_text
        pages_skipped += len(page_texts) - pages_with_text

        chunks_256: list[Document] = []
        for page_doc in pages:
            chunks_256.extend(chunker_256.chunk(page_doc))

        chunks_128: list[Document] = []
        for page_doc in pages:
            chunks_128.extend(chunker_128.chunk(page_doc))

        total_chunks_256 += len(chunks_256)
        total_chunks_128 += len(chunks_128)

        for chunk in chunks_256:
            n_tokens = len(enc.encode(chunk.text))
            all_token_counts.append(n_tokens)
            if chunk.page_start is not None:
                page_meta_coverage += 1

        print(f"  {name}: {pages_with_text} pages → {len(chunks_256)} chunks@256 / {len(chunks_128)} chunks@128")

    avg_tokens = statistics.mean(all_token_counts) if all_token_counts else 0
    p50_tokens = statistics.median(all_token_counts) if all_token_counts else 0
    p95_tokens = sorted(all_token_counts)[int(len(all_token_counts) * 0.95)] if len(all_token_counts) >= 20 else 0

    result = {
        "total_pages_parsed": total_pages,
        "pages_skipped_empty": pages_skipped,
        "total_chunks_256": total_chunks_256,
        "total_chunks_128": total_chunks_128,
        "chunk_ratio_256_vs_128": total_chunks_128 / max(total_chunks_256, 1),
        "avg_tokens_per_chunk": round(avg_tokens, 1),
        "p50_tokens_per_chunk": round(p50_tokens, 1),
        "p95_tokens_per_chunk": round(p95_tokens, 1),
        "page_metadata_coverage_pct": page_meta_coverage / max(total_chunks_256, 1) * 100,
    }
    print(f"  total pages parsed:        {result['total_pages_parsed']}")
    print(f"  chunks (chunk_size=256):   {result['total_chunks_256']}")
    print(f"  avg tokens/chunk:          {result['avg_tokens_per_chunk']:.1f}")
    print(f"  p50 / p95 tokens:          {result['p50_tokens_per_chunk']:.0f} / {result['p95_tokens_per_chunk']:.0f}")
    print(f"  page metadata coverage:    {result['page_metadata_coverage_pct']:.1f}%")
    return result


def eval_a2_api(tmpdir: Path) -> dict:
    """A2: FastAPI retrieve — latency distribution and concurrent throughput."""
    print("\n[A2] FastAPI /v1/retrieve performance")

    # Build a corpus from the synthetic QA set and index it with KeywordRetriever
    corpus_docs = []
    for i, qa in enumerate(SYNTHETIC_QA):
        corpus_docs.append(Document(
            doc_id=f"supp_{i}", text=qa["supp"], title=f"Fact {i}",
            page_start=i + 1, page_end=i + 1, source="corpus.pdf",
        ))
        corpus_docs.append(Document(
            doc_id=f"dist_{i}", text=qa["dist"], title=f"Distractor {i}",
            page_start=i + 1, page_end=i + 1, source="corpus.pdf",
        ))
    retriever = KeywordRetriever(corpus_docs)

    # Inject retriever into the API
    import importlib
    import src.api.server as server_mod
    importlib.reload(server_mod)
    server_mod._registry._registry["eval"] = retriever

    from fastapi.testclient import TestClient
    client = TestClient(server_mod.app)

    queries = [qa["q"] for qa in SYNTHETIC_QA]

    # ── Single-threaded latency baseline ────────────────────────────────────
    latencies_single: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        resp = client.post("/v1/retrieve", json={"query": q, "top_k": 5, "index_id": "eval"})
        latencies_single.append((time.perf_counter() - t0) * 1000)
        assert resp.status_code == 200

    # ── Concurrent load (8 workers, 4 repeats = 120 total requests) ─────────
    all_concurrent_latencies: list[float] = []
    errors = 0

    def _one_request(q: str) -> float:
        t0 = time.perf_counter()
        r = client.post("/v1/retrieve", json={"query": q, "top_k": 5, "index_id": "eval"})
        if r.status_code != 200:
            return -1.0
        return (time.perf_counter() - t0) * 1000

    concurrent_queries = queries * 4  # 120 requests
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_one_request, q) for q in concurrent_queries]
        for f in concurrent.futures.as_completed(futures):
            lat = f.result()
            if lat < 0:
                errors += 1
            else:
                all_concurrent_latencies.append(lat)
    t_total = time.perf_counter() - t_start
    throughput = len(concurrent_queries) / t_total

    def _pct(vals: list[float], p: float) -> float:
        vals_s = sorted(vals)
        idx = max(0, int(len(vals_s) * p) - 1)
        return round(vals_s[idx], 1)

    result = {
        "total_requests": len(concurrent_queries),
        "errors": errors,
        "error_rate_pct": errors / len(concurrent_queries) * 100,
        "throughput_rps": round(throughput, 1),
        "single_p50_ms": _pct(latencies_single, 0.50),
        "single_p95_ms": _pct(latencies_single, 0.95),
        "single_p99_ms": _pct(latencies_single, 0.99),
        "concurrent_p50_ms": _pct(all_concurrent_latencies, 0.50),
        "concurrent_p95_ms": _pct(all_concurrent_latencies, 0.95),
        "concurrent_p99_ms": _pct(all_concurrent_latencies, 0.99),
    }
    print(f"  single-thread  P50/P95/P99:  {_fmt_ms(result['single_p50_ms'])} / {_fmt_ms(result['single_p95_ms'])} / {_fmt_ms(result['single_p99_ms'])}")
    print(f"  concurrent(8)  P50/P95/P99:  {_fmt_ms(result['concurrent_p50_ms'])} / {_fmt_ms(result['concurrent_p95_ms'])} / {_fmt_ms(result['concurrent_p99_ms'])}")
    print(f"  throughput:                  {result['throughput_rps']:.1f} req/s  (errors={errors})")

    # ── Verify response structure ────────────────────────────────────────────
    resp = client.post("/v1/retrieve", json={"query": "capital of France", "top_k": 3, "index_id": "eval"})
    body = resp.json()
    first_passage = body["passages"][0] if body["passages"] else {}
    result["passage_fields_ok"] = all(k in first_passage for k in ("doc_id", "title", "text", "score", "rank", "page_start"))
    result["page_metadata_in_response"] = first_passage.get("page_start") is not None
    print(f"  response has page metadata:  {result['page_metadata_in_response']}")
    return result


def eval_a3_citation(scorer, use_real: bool) -> dict:
    """A3: NLI citation evaluation on the synthetic QA set."""
    mode = "real HHEM" if use_real else "overlap-stub"
    print(f"\n[A3] NLI citation evaluation  (scorer={mode})")

    evaluator = CitationEvaluator(scorer)

    # Build retriever over the full corpus
    corpus_docs: list[Document] = []
    doc_map: dict[str, Document] = {}
    for i, qa in enumerate(SYNTHETIC_QA):
        supp = Document(doc_id=f"supp_{i}", text=qa["supp"], title=f"Fact {i}",
                        page_start=i + 1, page_end=i + 1, source="corpus.pdf")
        dist = Document(doc_id=f"dist_{i}", text=qa["dist"], title=f"Distractor {i}",
                        page_start=i + 1, page_end=i + 1, source="corpus.pdf")
        corpus_docs.extend([supp, dist])
        doc_map[f"supp_{i}"] = supp
        doc_map[f"dist_{i}"] = dist

    retriever = KeywordRetriever(corpus_docs)

    attribution_rates: list[float] = []
    supporting_hits: list[bool] = []
    page_groundings: list[float] = []
    # Per-scenario: does the supporting doc appear in top-5?
    retrieval_hit: list[bool] = []
    # NLI: does the correct supporting passage score >= threshold?
    correct_passage_consistent: list[bool] = []
    # NLI on answer-only (single passage)
    answer_in_supp: list[float] = []
    answer_in_dist: list[float] = []

    for i, qa in enumerate(SYNTHETIC_QA):
        answer = qa["a"]
        retrieved = retriever.retrieve(qa["q"], top_k=5)
        retrieval_hit.append(any(d.doc_id == f"supp_{i}" for d in retrieved))

        citation = evaluator.evaluate(answer, retrieved)
        attribution_rates.append(citation.answer_attribution_rate)
        supporting_hits.append(citation.supporting_passage_hit)
        if citation.page_grounding_accuracy is not None:
            page_groundings.append(citation.page_grounding_accuracy)

        # Score supporting vs distractor passage independently
        supp_res = scorer.score(qa["supp"], answer)
        dist_res = scorer.score(qa["dist"], answer)
        correct_passage_consistent.append(supp_res.is_consistent)
        answer_in_supp.append(supp_res.score)
        answer_in_dist.append(dist_res.score)

    n = len(SYNTHETIC_QA)
    avg_supp_score = statistics.mean(answer_in_supp)
    avg_dist_score = statistics.mean(answer_in_dist)

    result = {
        "scorer_mode": mode,
        "num_queries": n,
        "retrieval_recall_at_5_pct": sum(retrieval_hit) / n * 100,
        "avg_attribution_rate": round(statistics.mean(attribution_rates), 3),
        "supporting_hit_rate_pct": sum(supporting_hits) / n * 100,
        "avg_page_grounding_accuracy": round(statistics.mean(page_groundings), 3) if page_groundings else None,
        "correct_passage_consistent_pct": sum(correct_passage_consistent) / n * 100,
        "avg_supp_nli_score": round(avg_supp_score, 3),
        "avg_dist_nli_score": round(avg_dist_score, 3),
        "supp_vs_dist_delta": round(avg_supp_score - avg_dist_score, 3),
    }
    print(f"  queries evaluated:           {n}")
    print(f"  retrieval recall@5:          {result['retrieval_recall_at_5_pct']:.1f}%")
    print(f"  avg attribution rate:        {result['avg_attribution_rate']:.3f}")
    print(f"  supporting hit rate:         {result['supporting_hit_rate_pct']:.1f}%")
    print(f"  correct passage consistent:  {result['correct_passage_consistent_pct']:.1f}%")
    print(f"  avg NLI score (supp/dist):   {result['avg_supp_nli_score']:.3f} / {result['avg_dist_nli_score']:.3f}  Δ={result['supp_vs_dist_delta']:+.3f}")
    if result["avg_page_grounding_accuracy"] is not None:
        print(f"  page grounding accuracy:     {result['avg_page_grounding_accuracy']:.3f}")
    return result


def eval_integration(tmpdir: Path, scorer) -> dict:
    """End-to-end: PDF ingest → docstore → retrieve → NLI score."""
    print("\n[Integration] PDF → ingest → retrieve → NLI")

    # Build a 5-page PDF with QA supporting texts
    qa_subset = SYNTHETIC_QA[:5]
    page_texts = [qa["supp"] for qa in qa_subset]
    pdf_bytes = _build_pdf_bytes(page_texts)
    pdf_path = tmpdir / "integration.pdf"
    pdf_path.write_bytes(pdf_bytes)

    # Ingest
    parser = PdfParser()
    chunker = TokenAwareChunker(chunk_size=256, overlap=32)
    pages = parser.parse(pdf_path)
    all_chunks: list[Document] = []
    for page_doc in pages:
        all_chunks.extend(chunker.chunk(page_doc))

    docstore_path = tmpdir / "integration.jsonl"
    offsets_path = tmpdir / "integration.offsets"
    save_docstore(docstore_path, all_chunks)
    build_docstore_offsets(docstore_path, offsets_path)
    reloaded = load_docstore(docstore_path)

    # Retrieve and score
    retriever = KeywordRetriever(reloaded)
    evaluator = CitationEvaluator(scorer)

    hits = 0
    for qa in qa_subset:
        retrieved = retriever.retrieve(qa["q"], top_k=3)
        citation = evaluator.evaluate(qa["a"], retrieved)
        if citation.supporting_passage_hit:
            hits += 1

    result = {
        "pages_ingested": len(pages),
        "chunks_produced": len(all_chunks),
        "docstore_reload_ok": len(reloaded) == len(all_chunks),
        "page_metadata_ok": all(d.page_start is not None for d in reloaded),
        "nli_hits_over_5_queries": hits,
        "nli_hit_rate_pct": hits / len(qa_subset) * 100,
    }
    print(f"  pages ingested:    {result['pages_ingested']}")
    print(f"  chunks produced:   {result['chunks_produced']}")
    print(f"  docstore reload:   {'OK' if result['docstore_reload_ok'] else 'FAIL'}")
    print(f"  page metadata:     {'OK' if result['page_metadata_ok'] else 'FAIL'}")
    print(f"  NLI hit rate:      {result['nli_hit_rate_pct']:.1f}%  ({hits}/{len(qa_subset)})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 full evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--use-real-hhem", action="store_true",
                   help="Load Vectara HHEM model for NLI scoring (~500MB download on first run).")
    p.add_argument("--hhem-model", default=HHEMScorer.MODEL_NAME,
                   help="HHEM model name (only used with --use-real-hhem).")
    p.add_argument("--output", type=Path, default=None,
                   help="Write JSON report to this path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Phase 2 Evaluation Report")
    print("=" * 60)

    if args.use_real_hhem:
        print("\nLoading HHEM model (may take a moment)...")
        scorer = HHEMScorer(model_name=args.hhem_model)
    else:
        print("\n[INFO] Using overlap-stub scorer (pass --use-real-hhem for real HHEM)")
        scorer = _OverlapScorer()

    report: dict[str, Any] = {"scorer": "real_hhem" if args.use_real_hhem else "overlap_stub"}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        report["A0_schema"] = eval_a0_schema(tmp)
        report["A1_ingestion"] = eval_a1_ingestion(tmp)
        report["A2_api"] = eval_a2_api(tmp)
        report["A3_citation"] = eval_a3_citation(scorer, use_real=args.use_real_hhem)
        report["integration"] = eval_integration(tmp, scorer)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    a0 = report["A0_schema"]
    a1 = report["A1_ingestion"]
    a2 = report["A2_api"]
    a3 = report["A3_citation"]
    ie = report["integration"]

    all_ok = (
        a0["new_fields_preserved_pct"] == 100.0
        and a0["old_docs_backward_compat_pct"] == 100.0
        and a0["lazy_docstore_random_access_accuracy"] == 100.0
        and a1["page_metadata_coverage_pct"] == 100.0
        and a2["error_rate_pct"] == 0.0
        and ie["docstore_reload_ok"]
        and ie["page_metadata_ok"]
    )

    print(f"  A0 backward compat:        {'PASS' if a0['old_docs_backward_compat_pct'] == 100 else 'FAIL'}")
    print(f"  A0 field preservation:     {'PASS' if a0['new_fields_preserved_pct'] == 100 else 'FAIL'}")
    print(f"  A1 page metadata 100%:     {'PASS' if a1['page_metadata_coverage_pct'] == 100 else 'FAIL'}")
    print(f"  A1 avg tokens/chunk:       {a1['avg_tokens_per_chunk']:.1f}")
    print(f"  A2 error rate:             {a2['error_rate_pct']:.1f}%  {'PASS' if a2['error_rate_pct']==0 else 'FAIL'}")
    print(f"  A2 throughput:             {a2['throughput_rps']:.1f} req/s")
    print(f"  A3 retrieval recall@5:     {a3['retrieval_recall_at_5_pct']:.1f}%")
    print(f"  A3 supp vs dist Δ:         {a3['supp_vs_dist_delta']:+.3f}  ({'scorer discriminates' if a3['supp_vs_dist_delta'] > 0.05 else 'low discrimination'})")
    print(f"  Integration NLI hit rate:  {ie['nli_hit_rate_pct']:.1f}%")
    print(f"\n  Overall: {'ALL CHECKS PASS' if all_ok else 'SOME CHECKS FAILED'}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        args.output.write_text(_json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
