"""Thread-safe registry that lazily loads retriever instances for each index.

An *index* is discovered as any sub-directory of the configured ``data_dir``
that contains a ``docstore.jsonl`` file.  Retriever type is auto-detected from
the files present:

- ``index.faiss`` + ``dense_config.json``  →  :class:`FaissDenseRetriever`
- ``bm25.pkl``                             →  :class:`BM25Retriever`

Dense takes priority when both are present.  The first ``get_retriever()`` call
for a given ``index_id`` acquires a per-index ``threading.Lock`` to avoid
duplicate initialisation under concurrent requests.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.logging_utils import get_logger
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss_dense import FaissDenseRetriever

logger = get_logger(__name__)


@dataclass
class _IndexEntry:
    retriever: Any
    index_type: str   # "dense" | "bm25"


class IndexRegistry:
    """Singleton-style registry; one instance is shared across all requests.

    Instantiation is cheap: no indexes are loaded until first access.

    Args:
        data_dir: Root directory that contains one sub-directory per index.
    """

    def __init__(self, data_dir: str | Path = "data/indexes") -> None:
        self._data_dir = Path(data_dir)
        self._entries: dict[str, _IndexEntry] = {}
        self._entry_locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()  # protects _entry_locks dict

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def available_index_ids(self) -> list[str]:
        """Return sorted list of discoverable index IDs (not necessarily loaded)."""
        if not self._data_dir.exists():
            return []
        return sorted(
            d.name
            for d in self._data_dir.iterdir()
            if d.is_dir() and (d / "docstore.jsonl").exists()
        )

    def loaded_index_ids(self) -> list[str]:
        """Return sorted list of index IDs whose retrievers are in memory."""
        return sorted(self._entries.keys())

    # ------------------------------------------------------------------ #
    # Retriever access (thread-safe lazy load)
    # ------------------------------------------------------------------ #

    def get_retriever(self, index_id: str) -> Any:
        """Return the retriever for *index_id*, loading it on first access."""
        if index_id not in self._entries:
            self._load(index_id)
        return self._entries[index_id].retriever

    def index_type(self, index_id: str) -> str:
        """Return the retriever type string for *index_id* (loads if needed)."""
        if index_id not in self._entries:
            self._load(index_id)
        return self._entries[index_id].index_type

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    def _load(self, index_id: str) -> None:
        # Ensure a per-index lock exists (global_lock protects this dict write)
        with self._global_lock:
            if index_id not in self._entry_locks:
                self._entry_locks[index_id] = threading.Lock()

        with self._entry_locks[index_id]:
            if index_id in self._entries:
                return  # another thread finished loading while we waited

            index_dir = self._data_dir / index_id
            if not index_dir.is_dir():
                raise KeyError(
                    f"Index {index_id!r} not found "
                    f"(looked in {self._data_dir.resolve()})"
                )
            if not (index_dir / "docstore.jsonl").exists():
                raise KeyError(
                    f"Index {index_id!r}: missing docstore.jsonl in {index_dir}"
                )

            logger.info("Loading index %r from %s", index_id, index_dir)
            retriever, itype = _build_retriever(index_dir)
            self._entries[index_id] = _IndexEntry(retriever=retriever, index_type=itype)
            logger.info("Index %r loaded (type=%s)", index_id, itype)


def _build_retriever(index_dir: Path) -> tuple[Any, str]:
    """Auto-detect index type from files present and return ``(retriever, type_str)``."""
    has_dense = (
        (index_dir / "index.faiss").exists()
        and (index_dir / "dense_config.json").exists()
    )
    has_bm25 = (index_dir / "bm25.pkl").exists()

    if has_dense:
        offsets = index_dir / "docstore.offsets"
        retriever = FaissDenseRetriever(
            faiss_index_path=index_dir / "index.faiss",
            docstore_path=index_dir / "docstore.jsonl",
            dense_config_path=index_dir / "dense_config.json",
            docstore_offsets_path=offsets if offsets.exists() else None,
        )
        return retriever, "dense"

    if has_bm25:
        retriever = BM25Retriever(
            bm25_path=index_dir / "bm25.pkl",
            docstore_path=index_dir / "docstore.jsonl",
        )
        return retriever, "bm25"

    raise ValueError(
        f"Cannot determine retriever type for {index_dir}: "
        "need index.faiss+dense_config.json (dense) or bm25.pkl (BM25)."
    )
