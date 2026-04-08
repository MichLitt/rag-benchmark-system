"""Post-hoc NLI attribution metrics for PDF Q&A evaluation.

All three metrics are computed from a single (passages × sentences) NLI score
matrix, so the model is called exactly once per (passage, sentence) pair.

Metric definitions (per Phase 2 plan §A3)
------------------------------------------
answer_attribution_rate
    Fraction of answer sentences that have ≥ 1 retrieved passage with
    entailment_prob ≥ threshold.  Range [0, 1].  Primary metric.

supporting_passage_hit
    Fraction of the top-k retrieved passages that support ≥ 1 answer
    sentence (entailment_prob ≥ threshold).  Range [0, 1].

page_grounding_accuracy
    Among the supporting passages, fraction whose page range (expanded by
    ±page_tolerance pages) intersects the gold page set.  Range [0, 1].
    ``None`` when no gold pages are available or when no passage is
    supporting.

Notes
-----
- Sentence splitting uses ``nltk.sent_tokenize`` with a regex fallback.
- Samples without a gold_page_set yield ``page_grounding_accuracy = None``
  and are excluded from aggregate means.
- The scorer is injected (accepts any :class:`NLIScorer` duck-type), making
  unit tests fast without downloading the HHEM model.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.hhem_scorer import NLIScorer

from src.types import Document

logger = logging.getLogger(__name__)

NLI_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences.

    Tries ``nltk.sent_tokenize``; downloads punkt data on first use.
    Falls back to a simple regex splitter if nltk is unavailable.
    """
    text = text.strip()
    if not text:
        return []
    try:
        import nltk  # noqa: PLC0415
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            # punkt data not yet downloaded — grab it silently
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)
        return [s for s in sentences if s.strip()]
    except ImportError:
        # Regex fallback: split on ". " / "! " / "? "
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Page-range helper
# ---------------------------------------------------------------------------

def _page_range_intersects(
    page_start: int | None,
    page_end: int | None,
    gold_pages: set[int],
    tolerance: int = 1,
) -> bool:
    """Return ``True`` if the passage's expanded page range overlaps *gold_pages*.

    The passage pages are expanded by ±*tolerance* before the intersection
    test, implementing the ±1 page tolerance required by the plan spec.

    Args:
        page_start: First page of the passage chunk (1-indexed).
        page_end: Last page of the passage chunk (1-indexed).
        gold_pages: Set of annotated gold page numbers.
        tolerance: Pages to add on each side of [page_start, page_end].
    """
    if page_start is None or page_end is None or not gold_pages:
        return False
    expanded = set(range(page_start - tolerance, page_end + tolerance + 1))
    return bool(expanded & gold_pages)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CitationResult:
    """Per-query NLI citation evaluation output.

    Metric values may be ``None`` when inputs are insufficient (e.g. no gold
    pages for page_grounding_accuracy, or empty answer/passages).
    """

    answer_attribution_rate: float | None
    """Fraction of answer sentences attributed to ≥ 1 passage."""

    supporting_passage_hit: float | None
    """Fraction of passages that support ≥ 1 answer sentence."""

    page_grounding_accuracy: float | None
    """Among supporting passages, fraction on the correct pages. ``None`` when
    gold_page_set is empty or no passages are supporting."""

    # Diagnostic counts (useful for debugging and aggregate reporting)
    attributed_sentence_count: int = 0
    total_sentence_count: int = 0
    supporting_passage_count: int = 0
    total_passage_count: int = 0
    grounded_supporting_count: int = 0


# ---------------------------------------------------------------------------
# CitationEvaluator
# ---------------------------------------------------------------------------

class CitationEvaluator:
    """Compute NLI-based citation metrics for a single QA example.

    Builds a (passages × sentences) NLI score matrix once and derives all
    three metrics, avoiding redundant model inference.

    Args:
        scorer: Any object with ``score(premise, hypothesis) -> float``.
                Typically :class:`HHEMScorer`; a lightweight mock is used in
                tests to avoid downloading the model.
        threshold: Entailment probability cutoff (default 0.5, plan §A3).
        page_tolerance: Page-range expansion for page_grounding_accuracy
                        (default 1, i.e. ±1 page tolerance).
    """

    def __init__(
        self,
        scorer: NLIScorer,
        threshold: float = NLI_THRESHOLD,
        page_tolerance: int = 1,
    ) -> None:
        self._scorer = scorer
        self._threshold = threshold
        self._page_tolerance = page_tolerance

    def evaluate(
        self,
        answer: str,
        passages: list[Document],
        gold_page_set: set[int] | None = None,
    ) -> CitationResult:
        """Evaluate citation quality for one generated answer.

        Args:
            answer: Generated answer text (may contain multiple sentences).
            passages: Retrieved top-k passages with page metadata.
            gold_page_set: Gold page numbers from the PDF annotation.
                           Required for page_grounding_accuracy; metric is
                           ``None`` when absent.

        Returns:
            :class:`CitationResult` with all three metrics populated.
        """
        # Guard: empty answer
        if not answer.strip():
            return CitationResult(
                answer_attribution_rate=None,
                supporting_passage_hit=None,
                page_grounding_accuracy=None,
            )

        sentences = _split_sentences(answer)

        # Guard: degenerate inputs
        if not sentences or not passages:
            return CitationResult(
                answer_attribution_rate=0.0 if sentences else None,
                supporting_passage_hit=0.0 if passages else None,
                page_grounding_accuracy=None,
                total_sentence_count=len(sentences),
                total_passage_count=len(passages),
            )

        n_passages = len(passages)
        n_sentences = len(sentences)

        # ---- Build NLI score matrix (n_passages × n_sentences) ----
        # scores[p][s] = entailment_prob(passage_p.text, sentence_s)
        scores: list[list[float]] = [
            [self._scorer.score(p.text, s) for s in sentences]
            for p in passages
        ]

        # ---- answer_attribution_rate ----
        # A sentence is "attributed" if any passage exceeds the threshold
        attributed = sum(
            1
            for s_idx in range(n_sentences)
            if any(
                scores[p_idx][s_idx] >= self._threshold
                for p_idx in range(n_passages)
            )
        )
        attribution_rate = attributed / n_sentences

        # ---- supporting_passage_hit ----
        # A passage is "supporting" if it exceeds the threshold for any sentence
        supporting_flags: list[bool] = [
            any(
                scores[p_idx][s_idx] >= self._threshold
                for s_idx in range(n_sentences)
            )
            for p_idx in range(n_passages)
        ]
        supporting_count = sum(supporting_flags)
        passage_hit = supporting_count / n_passages

        # ---- page_grounding_accuracy ----
        grounded = 0
        page_accuracy: float | None = None
        if gold_page_set:
            grounded = sum(
                1
                for p_idx, (passage, is_supporting) in enumerate(
                    zip(passages, supporting_flags)
                )
                if is_supporting
                and _page_range_intersects(
                    passage.page_start,
                    passage.page_end,
                    gold_page_set,
                    self._page_tolerance,
                )
            )
            page_accuracy = (
                grounded / supporting_count if supporting_count > 0 else None
            )

        logger.debug(
            "CitationEval: %d/%d sentences attributed, %d/%d passages supporting",
            attributed, n_sentences, supporting_count, n_passages,
        )

        return CitationResult(
            answer_attribution_rate=attribution_rate,
            supporting_passage_hit=passage_hit,
            page_grounding_accuracy=page_accuracy,
            attributed_sentence_count=attributed,
            total_sentence_count=n_sentences,
            supporting_passage_count=supporting_count,
            total_passage_count=n_passages,
            grounded_supporting_count=grounded,
        )
