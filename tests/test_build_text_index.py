from __future__ import annotations

import json

from scripts.build_text_index import build_text_index
from src.api.index_registry import IndexRegistry


def test_build_text_index_creates_retrievable_bm25_index(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "guide.md").write_text("# Release guide\nA candidate needs a promoted gate.", encoding="utf-8")
    manifest = workspace / "corpus.json"
    manifest.write_text(
        json.dumps({"index_name": "g3-test", "documents": ["guide.md"]}),
        encoding="utf-8",
    )
    index_dir = tmp_path / "indexes" / "g3-test"

    result = build_text_index(
        manifest_path=manifest,
        workspace_root=workspace,
        output_dir=index_dir,
        chunk_size=64,
        overlap=8,
    )

    assert result["doc_count"] > 0
    assert (index_dir / "docstore.jsonl").is_file()
    assert (index_dir / "bm25.pkl").is_file()
    assert (index_dir / "corpus-build-manifest.json").is_file()
    retriever = IndexRegistry(data_dir=tmp_path / "indexes").get_retriever("g3-test")
    assert retriever.retrieve("promoted gate", top_k=1)[0].source == "guide.md"
