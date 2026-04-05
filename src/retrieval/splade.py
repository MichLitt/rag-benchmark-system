"""SPLADE sparse retriever.

Uses ``naver-splade/splade-cocondenser-ensembledistil`` to encode documents
and queries as learned sparse representations, then retrieves via dot-product
scoring over a prebuilt scipy CSR sparse index matrix.

Index files on disk
-------------------
- ``splade_index.npz``:   scipy CSR matrix, shape ``(n_docs, vocab_size)``, float32.
- ``splade_config.json``: ``{"model_name": "...", "vocab_size": N, "num_docs": N}``.

SPLADE aggregation formula
--------------------------
For each token position *t* and vocabulary dimension *v*:

    activation[v] = log(1 + ReLU(MLM_logits[t, v]))

The per-document sparse vector is the **max** over all token positions::

    doc_vec[v] = max_t( log(1 + ReLU(logits[t, v])) )

Query encoding uses the same formula.  Relevance = dot product(query_vec, doc_vec).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, TYPE_CHECKING, runtime_checkable

import numpy as np

from src.retrieval.docstore import load_docstore
from src.types import Document

if TYPE_CHECKING:
    import scipy.sparse

logger = logging.getLogger(__name__)

SPLADE_MODEL_NAME = "naver-splade/splade-cocondenser-ensembledistil"


# ---------------------------------------------------------------------------
# SparseEncoder protocol — keeps SPLADERetriever testable without a real model
# ---------------------------------------------------------------------------

@runtime_checkable
class SparseEncoder(Protocol):
    """Structural protocol for SPLADE-style sparse text encoders."""

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (= number of columns in the index matrix)."""
        ...

    def encode(self, text: str) -> np.ndarray:
        """Return a float32 array of shape ``(vocab_size,)``."""
        ...

    def encode_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> "scipy.sparse.csr_matrix":
        """Return a sparse CSR matrix of shape ``(len(texts), vocab_size)``."""
        ...


# ---------------------------------------------------------------------------
# HFSpladeEncoder — wraps the HuggingFace SPLADE MLM model
# ---------------------------------------------------------------------------

class HFSpladeEncoder:
    """HuggingFace SPLADE encoder.

    Loads an MLM model (e.g. ``naver-splade/splade-cocondenser-ensembledistil``)
    and encodes texts using the SPLADE max-pool + log1p(relu) aggregation.

    Args:
        model_name: HuggingFace model ID.
        device: ``"cpu"`` or ``"cuda"``; auto-detected when ``None``.
    """

    def __init__(
        self,
        model_name: str = SPLADE_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        logger.info("Loading SPLADE encoder %r on %s …", model_name, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)
        self._model.to(device)
        self._model.eval()
        self._vocab_size_val: int = self._model.config.vocab_size
        logger.info("SPLADE encoder ready: vocab_size=%d", self._vocab_size_val)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size_val

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text → ``(vocab_size,)`` float32 SPLADE vector."""
        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=False
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits  # (1, seq_len, vocab_size)
        vec = torch.log1p(torch.relu(logits)).max(dim=1).values.squeeze(0)
        return vec.cpu().numpy().astype(np.float32)

    def encode_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> "scipy.sparse.csr_matrix":
        """Encode a list of texts → sparse ``(n_texts, vocab_size)`` CSR matrix."""
        import scipy.sparse
        import torch

        if not texts:
            return scipy.sparse.csr_matrix((0, self._vocab_size_val), dtype=np.float32)

        rows: list[scipy.sparse.csr_matrix] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits  # (B, seq_len, vocab)
            # Mask padded positions before max-pooling
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, seq_len, 1)
            masked = logits * mask - 1e9 * (1.0 - mask)
            vecs = torch.log1p(torch.relu(masked)).max(dim=1).values  # (B, vocab)
            rows.append(
                scipy.sparse.csr_matrix(vecs.cpu().numpy().astype(np.float32))
            )
        return scipy.sparse.vstack(rows)


# ---------------------------------------------------------------------------
# SPLADERetriever
# ---------------------------------------------------------------------------

class SPLADERetriever:
    """Sparse retriever backed by a prebuilt SPLADE index.

    Accepts an explicit ``index_matrix`` and ``encoder`` so tests can inject
    lightweight fakes without loading the real model.  Production code uses
    :meth:`from_disk`.

    Args:
        index_matrix: CSR matrix of shape ``(n_docs, vocab_size)``, float32.
        docs: Document list parallel to ``index_matrix`` rows.
        encoder: Encoder used to convert query strings to SPLADE vectors.
    """

    def __init__(
        self,
        index_matrix: "scipy.sparse.csr_matrix",
        docs: list[Document],
        encoder: SparseEncoder,
    ) -> None:
        self._index = index_matrix
        self._docs = docs
        self._encoder = encoder

    @classmethod
    def from_disk(
        cls,
        splade_index_path: str | Path,
        docstore_path: str | Path,
        splade_config_path: str | Path,
        model_name: str = SPLADE_MODEL_NAME,
        device: str | None = None,
    ) -> "SPLADERetriever":
        """Load a prebuilt SPLADE index from disk and return a ready retriever."""
        import scipy.sparse

        matrix = scipy.sparse.load_npz(str(splade_index_path)).tocsr()
        docs = load_docstore(docstore_path)

        with Path(splade_config_path).open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        used_model = str(cfg.get("model_name", model_name))
        encoder = HFSpladeEncoder(model_name=used_model, device=device)

        logger.info(
            "SPLADERetriever loaded: %d docs, index shape=%s, nnz=%d",
            len(docs), matrix.shape, matrix.nnz,
        )
        return cls(matrix, docs, encoder)

    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        docs, _ = self.retrieve_with_scores(query, top_k)
        return docs

    def retrieve_with_scores(
        self, query: str, top_k: int
    ) -> tuple[list[Document], list[float]]:
        """Return top-k documents and their SPLADE dot-product scores."""
        if top_k <= 0 or not self._docs:
            return [], []

        q_vec = self._encoder.encode(query)  # (vocab_size,)
        # Sparse dot product: (n_docs, vocab) @ (vocab,) → (n_docs,)
        scores = np.asarray(self._index.dot(q_vec)).flatten()

        n = len(self._docs)
        k = min(top_k, n)
        if k < n:
            top_idx = np.argpartition(-scores, k - 1)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        else:
            top_idx = np.argsort(-scores)

        result_docs = [self._docs[int(i)] for i in top_idx]
        result_scores = [float(scores[int(i)]) for i in top_idx]
        return result_docs, result_scores


# ---------------------------------------------------------------------------
# Index-building utility (used by scripts/build_splade_index.py)
# ---------------------------------------------------------------------------

def build_splade_index(
    docs: list[Document],
    encoder: SparseEncoder,
    *,
    batch_size: int = 32,
) -> "scipy.sparse.csr_matrix":
    """Encode *docs* and return a CSR sparse index matrix.

    Args:
        docs: Documents to encode.
        encoder: Any :class:`SparseEncoder` (real or fake).
        batch_size: Documents per encoding batch.

    Returns:
        CSR matrix of shape ``(len(docs), vocab_size)``, float32.
    """
    import scipy.sparse

    if not docs:
        return scipy.sparse.csr_matrix((0, encoder.vocab_size), dtype=np.float32)

    texts = [f"{doc.title}\n{doc.text}".strip() for doc in docs]
    matrix = encoder.encode_batch(texts, batch_size=batch_size)
    logger.info(
        "Built SPLADE index: %d docs, shape=%s, nnz=%d",
        len(docs), matrix.shape, matrix.nnz,
    )
    return matrix
