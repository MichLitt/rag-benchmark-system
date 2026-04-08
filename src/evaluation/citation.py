from __future__ import annotations

from dataclasses import dataclass, field

from src.evaluation.hhem_scorer import HHEMScorer
from src.types import Document


@dataclass(frozen=True)
class CitationResult:
    answer_attribution_rate: float
    """Fraction of retrieved passages that are consistent with the answer (NLI-based).

    Range [0, 1]. A passage is counted as consistent if its HHEM score >= threshold.
    This is a retrieval grounding metric, not an LLM citation parser — it measures
    whether the answer *could* be attributed to the retrieved context.
    """

    supporting_passage_hit: bool
    """True if at least one retrieved passage is consistent with the answer."""

    page_grounding_accuracy: float | None
    """Of the consistent passages, what fraction carry page_start metadata.

    None when no consistent passages exist or no passages have page metadata.
    Only meaningful for PDF-ingested Documents with page_start/page_end set.
    """

    passage_scores: list[float] = field(default_factory=list)
    """Per-passage HHEM consistency scores, in the same order as the input passages."""


class CitationEvaluator:
    """Post-hoc NLI attribution evaluator.

    Evaluates whether a generated answer is attributable to retrieved passages
    using HHEM as the NLI backbone. Computes three metrics:

    - answer_attribution_rate: fraction of passages consistent with the answer
    - supporting_passage_hit: any passage above the consistency threshold
    - page_grounding_accuracy: fraction of consistent passages with page metadata

    This does not parse LLM output for citation markers — it is a post-hoc
    analysis that treats each retrieved passage as a potential source.
    """

    def __init__(self, scorer: HHEMScorer) -> None:
        self._scorer = scorer

    def evaluate(
        self,
        answer: str,
        passages: list[Document],
    ) -> CitationResult:
        """Evaluate citation grounding of an answer against retrieved passages.

        Args:
            answer: The generated answer string.
            passages: Retrieved Documents to score against the answer.

        Returns:
            CitationResult with attribution metrics.
        """
        if not passages or not answer.strip():
            return CitationResult(
                answer_attribution_rate=0.0,
                supporting_passage_hit=False,
                page_grounding_accuracy=None,
                passage_scores=[],
            )

        results = self._scorer.score_batch(
            [(doc.text, answer) for doc in passages]
        )

        scores = [r.score for r in results]
        consistent_flags = [r.is_consistent for r in results]

        attribution_rate = sum(consistent_flags) / len(consistent_flags)
        hit = any(consistent_flags)

        consistent_passages = [
            doc for doc, flag in zip(passages, consistent_flags) if flag
        ]
        page_grounding: float | None = None
        if consistent_passages:
            with_page = sum(
                1 for doc in consistent_passages if doc.page_start is not None
            )
            page_grounding = with_page / len(consistent_passages)

        return CitationResult(
            answer_attribution_rate=attribution_rate,
            supporting_passage_hit=hit,
            page_grounding_accuracy=page_grounding,
            passage_scores=scores,
        )
