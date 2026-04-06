"""Setwise LLM reranker (B2 — Stage 2 of two-stage reranking).

Presents a batch of ≤ *max_candidates* passages to an LLM and asks it to rank
them by relevance.  The LLM outputs a comma-separated list of 1-based indices
which is parsed into a document ordering.

When more passages are provided than *max_candidates*, they are partitioned into
overlapping sliding windows; each document's **best rank** across all windows is
used for the final merge.

Prompt format
-------------
::

    Given the query: "{query}"

    Rank the following passages from most to least relevant.
    Output ONLY a comma-separated list of passage indices (e.g., "3,1,5,2,4"):

    [1] {title}
    {text excerpt}

    [2] ...

Scores
------
The score assigned to each output document is ``1 / (rank + 1)`` (rank starts
at 0), matching the RRF reciprocal-rank style.  This keeps scores comparable
across different *top_k* values.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

from src.types import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TextGenerator protocol — injected for testability
# ---------------------------------------------------------------------------

@runtime_checkable
class TextGenerator(Protocol):
    """Structural protocol: any callable-style object with a ``generate`` method."""

    def generate(self, prompt: str) -> str:
        """Generate and return a text completion for *prompt*.

        The implementation may call an LLM API, a local model, or any other
        text-generation backend.  The return value should be the new text only
        (not the prompt itself).
        """
        ...


# ---------------------------------------------------------------------------
# HFTextGenerator — wraps a local HuggingFace causal LM (e.g. Qwen2.5-3B)
# ---------------------------------------------------------------------------

class HFTextGenerator:
    """Local HuggingFace text generator for setwise reranking.

    Loads ``model_name`` with ``AutoModelForCausalLM`` and exposes a
    ``generate`` method compatible with :class:`TextGenerator`.

    Args:
        model_name: HuggingFace model ID (default: ``Qwen/Qwen2.5-3B-Instruct``).
        device: ``"cpu"`` or ``"cuda"``; auto-detected when ``None``.
        max_new_tokens: Maximum tokens to generate per call (keep small for
            ranking tasks — the expected output is a short index list).
    """

    _DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
        max_new_tokens: int = 64,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._max_new_tokens = max_new_tokens

        logger.info("Loading setwise LLM %r on %s …", model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        logger.info("Setwise LLM ready.")

    def generate(self, prompt: str) -> str:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Return only the newly generated tokens
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# SetwiseLLMReranker
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    'Given the query: "{query}"\n\n'
    "Rank the following passages from most to least relevant.\n"
    'Output ONLY a comma-separated list of passage indices (e.g., "3,1,5,2,4"):\n\n'
    "{passages}"
)

_PASSAGE_SNIPPET_CHARS = 500


class SetwiseLLMReranker:
    """LLM-based setwise reranker.

    Presents at most *max_candidates* passages per LLM call.  If more passages
    are given, a sliding-window tournament is used: each doc's **best rank** over
    all windows it appears in determines the merged order.

    Args:
        generator: Any object satisfying the :class:`TextGenerator` protocol.
        max_candidates: Maximum passages per LLM call (plan specifies ≤ 10 to
            avoid context overrun in 3B-parameter models).
    """

    def __init__(self, generator: TextGenerator, max_candidates: int = 10) -> None:
        if max_candidates <= 0:
            raise ValueError(f"max_candidates must be > 0, got {max_candidates}")
        self._gen = generator
        self._max = max_candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        reranked, _ = self.rerank_with_scores(query, docs, top_k)
        return reranked

    def rerank_with_scores(
        self, query: str, docs: list[Document], top_k: int
    ) -> tuple[list[Document], list[float]]:
        """Rerank *docs* and return ``(documents, scores)`` where score = 1/(rank+1)."""
        if not docs or top_k <= 0:
            return [], []

        n = len(docs)
        if n <= self._max:
            order = self._rerank_window(query, docs)
        else:
            order = self._sliding_window_rank(query, docs)

        k = min(top_k, len(order))
        selected = [docs[i] for i in order[:k]]
        scores = [1.0 / (rank + 1) for rank in range(k)]
        return selected, scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_passages(self, docs: list[Document]) -> str:
        lines: list[str] = []
        for i, doc in enumerate(docs, start=1):
            snippet = f"{doc.title}\n{doc.text}".strip()[:_PASSAGE_SNIPPET_CHARS]
            lines.append(f"[{i}] {snippet}")
        return "\n\n".join(lines)

    def _parse_ranking(self, response: str, n: int) -> list[int]:
        """Parse LLM response into a complete 0-based index list.

        Extracts all integers from *response*, converts to 0-based, deduplicates,
        then appends any missing indices in their original order as a fallback.
        """
        nums = re.findall(r"\d+", response)
        seen: set[int] = set()
        result: list[int] = []
        for s in nums:
            idx = int(s) - 1  # 1-based → 0-based
            if 0 <= idx < n and idx not in seen:
                seen.add(idx)
                result.append(idx)
        # Append any indices the LLM omitted, in original order
        for i in range(n):
            if i not in seen:
                result.append(i)
        return result

    def _rerank_window(self, query: str, docs: list[Document]) -> list[int]:
        """Rerank a single window of ≤ max_candidates docs. Returns 0-based indices."""
        passages_text = self._format_passages(docs)
        prompt = _PROMPT_TEMPLATE.format(query=query, passages=passages_text)
        try:
            response = self._gen.generate(prompt)
        except Exception:  # noqa: BLE001
            logger.warning("TextGenerator.generate() failed; using original order.")
            response = ""
        return self._parse_ranking(response, len(docs))

    def _sliding_window_rank(self, query: str, docs: list[Document]) -> list[int]:
        """Merge rankings from overlapping windows.

        Window stride = max_candidates // 2 (50 % overlap).
        Each document's score = 1 / (best_rank + 1), where best_rank is the
        minimum (= best) rank the document achieved across all windows it appeared
        in.  Ties broken by original index.
        """
        n = len(docs)
        # initialise with worst possible rank (original index as fallback)
        best_rank: dict[int, float] = {i: float(n) for i in range(n)}

        step = max(1, self._max // 2)
        for start in range(0, n, step):
            window_idxs = list(range(start, min(start + self._max, n)))
            window_docs = [docs[i] for i in window_idxs]
            local_order = self._rerank_window(query, window_docs)
            for rank, local_pos in enumerate(local_order):
                global_idx = window_idxs[local_pos]
                if rank < best_rank[global_idx]:
                    best_rank[global_idx] = float(rank)

        return sorted(range(n), key=lambda i: (best_rank[i], i))
