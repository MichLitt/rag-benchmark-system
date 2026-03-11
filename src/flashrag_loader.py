from __future__ import annotations

import json
from pathlib import Path

from src.types import QuerySample


def load_flashrag_qa(path: str | Path, max_queries: int | None = None) -> list[QuerySample]:
    rows: list[QuerySample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            answers = row.get("golden_answers") or []
            if not answers and row.get("gold_answer"):
                answers = [row["gold_answer"]]
            sample = QuerySample(
                query_id=str(row.get("id", "")),
                question=str(row.get("question", "")),
                answers=[str(a) for a in answers],
                gold_doc_id=None,
                gold_titles=[str(t) for t in row.get("gold_titles", [])],
            )
            rows.append(sample)
            if max_queries is not None and max_queries > 0 and len(rows) >= max_queries:
                break
    return rows
