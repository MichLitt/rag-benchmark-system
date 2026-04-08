from __future__ import annotations

import threading
from typing import Any

from src.config import load_yaml_config
from src.retrieval.factory import build_retriever


class IndexRegistry:
    """Thread-safe registry mapping index_id → retriever instance.

    Indexes are loaded from YAML config files at startup via
    load_from_config(). The registry dict is protected by a threading.Lock;
    individual retriever read operations (FAISS search, LazyDocstore.get)
    are independently thread-safe after the A0 docstore fix.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registry: dict[str, Any] = {}

    def load_from_config(self, index_id: str, config_path: str) -> None:
        """Build a retriever from a YAML config and register it under index_id.

        Replaces any existing retriever registered under that id.
        Only disk-backed retrieval modes are supported (bm25, dense,
        dense_sharded, hybrid). The keyword mode requires a corpus list
        and is not suitable for the API.
        """
        cfg = load_yaml_config(config_path)
        # corpus=[] is fine for all disk-backed retriever modes
        retriever = build_retriever(cfg, corpus=[])
        with self._lock:
            old = self._registry.get(index_id)
            if old is not None and hasattr(old, "close"):
                try:
                    old.close()
                except Exception:
                    pass
            self._registry[index_id] = retriever

    def get(self, index_id: str) -> Any | None:
        with self._lock:
            return self._registry.get(index_id)

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._registry.keys())

    def close_all(self) -> None:
        with self._lock:
            for retriever in self._registry.values():
                if hasattr(retriever, "close"):
                    try:
                        retriever.close()
                    except Exception:
                        pass
            self._registry.clear()
