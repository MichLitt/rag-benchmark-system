"""Citation-constrained generation wrapper.

Wraps any ``GeneratorLike`` to:

1. Inject a citation instruction into the base generator's ``system_prompt``
   so the model outputs inline ``[N]`` references.
2. After each ``generate()`` call, parse the cited passage indices.
3. Score citation precision using an NLI scorer (HHEM by default): for each
   in-range cited passage, check whether ``scorer.score(passage, answer) >=
   threshold``.  Precision = supported / in_range_cited.
4. Expose ``last_citation_result`` so ``pipeline.py`` can read it without
   depending on this class directly (duck-typing / ``hasattr`` pattern).

The ``GeneratorLike`` protocol is fully satisfied — ``generate()`` has an
identical signature to the base generator.

Key design decision: HHEM is called once per cited passage with the **full
answer** as hypothesis (not sentence-by-sentence).  This keeps cost linear in
the number of citations (typically 1–3) rather than O(passages × sentences).
The full sentence-matrix NLI evaluation remains in ``src/evaluation/citation.py``
for post-hoc use.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.generation.base import GenerationResult
from src.logging_utils import get_logger
from src.types import Document

if TYPE_CHECKING:
    from src.evaluation.hhem_scorer import NLIScorer

logger = get_logger(__name__)

# Matches [1], [2,3], [1, 2, 3], [12] — passage indices as formatted by
# _format_contexts() in openai_compatible.py.
_CITATION_PATTERN: re.Pattern[str] = re.compile(
    r"\[(\d+(?:\s*,\s*\d+)*)\]"
)

NLI_THRESHOLD: float = 0.5


@dataclass
class CitationScoringResult:
    """Output of per-answer citation precision scoring.

    Attributes:
        citation_count: Number of unique passage indices cited in the answer
            (regardless of whether they are in-range).
        citation_precision: Fraction of in-range cited passages with
            ``hhem_score >= threshold``.  ``None`` when the scorer is
            disabled (``scorer_disabled=True``) or when no passages were
            cited.
        cited_indices: Sorted, unique, 1-based passage indices parsed from the
            answer text.
        hhem_scores: Mapping from 1-based index → HHEM score for in-range
            cited passages only.
        scorer_disabled: ``True`` when HHEM scoring was skipped
            (``score_citations=false`` in config).
    """

    citation_count: int
    citation_precision: float | None
    cited_indices: list[int]
    hhem_scores: dict[int, float] = field(default_factory=dict)
    scorer_disabled: bool = False


def parse_citation_indices(text: str) -> list[int]:
    """Extract unique, sorted 1-based passage indices from answer text.

    Handles ``[1]``, ``[2,3]``, ``[1, 2, 3]``.  Out-of-range indices are kept
    (filtered in :func:`score_citation_precision`).  Returns empty list when
    no citations are found.

    Args:
        text: Raw generated answer possibly containing ``[N]`` citations.

    Returns:
        Sorted list of unique 1-based indices.
    """
    indices: set[int] = set()
    for match in _CITATION_PATTERN.finditer(text):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                indices.add(int(part))
    return sorted(indices)


def score_citation_precision(
    answer: str,
    contexts: list[Document],
    scorer: "NLIScorer",
    threshold: float = NLI_THRESHOLD,
) -> CitationScoringResult:
    """Score citation precision for a generated answer.

    For each cited passage index that is in-range (1..len(contexts)), runs
    ``scorer.score(passage_text, answer)``.  Citation precision = supported /
    in_range_count.

    Args:
        answer: Raw answer text possibly containing ``[N]`` citations.
        contexts: Passages passed to the generator (1-indexed).
        scorer: Any :class:`~src.evaluation.hhem_scorer.NLIScorer`-compatible
            object.
        threshold: HHEM score cutoff for "supports" classification (default
            0.5).

    Returns:
        :class:`CitationScoringResult` with precision and per-passage scores.
    """
    cited = parse_citation_indices(answer)
    n_contexts = len(contexts)

    in_range = [i for i in cited if 1 <= i <= n_contexts]

    if not in_range:
        return CitationScoringResult(
            citation_count=len(cited),
            citation_precision=0.0 if cited else None,
            cited_indices=cited,
            hhem_scores={},
        )

    hhem_scores: dict[int, float] = {}
    for idx in in_range:
        passage = contexts[idx - 1]  # convert to 0-based
        try:
            hhem_scores[idx] = scorer.score(passage.text, answer)
        except Exception as exc:
            logger.warning("HHEM scoring failed for passage %d: %s", idx, exc)
            hhem_scores[idx] = 0.0

    supported = sum(1 for s in hhem_scores.values() if s >= threshold)
    precision = supported / len(in_range)

    return CitationScoringResult(
        citation_count=len(cited),
        citation_precision=precision,
        cited_indices=cited,
        hhem_scores=hhem_scores,
    )


class CitationConstrainedGenerator:
    """Wraps a base generator to enforce citation constraints and score precision.

    Responsibilities:

    1. **Citation injection**: Patches the wrapped generator's ``system_prompt``
       attribute with :data:`~src.generation.prompts.CITATION_INSTRUCTION` at
       construction time, so the model is instructed to output ``[N]``
       references without changing the ``generate()`` call site.

    2. **Post-generation parsing**: After each ``generate()`` call, parses
       ``[N]`` references from the answer text.

    3. **NLI scoring**: If a scorer is provided, runs HHEM on each cited
       in-range passage against the full answer.  One HHEM call per citation
       (not per sentence), keeping latency linear in citation count.

    4. **Result exposure**: Stores the most recent
       :class:`CitationScoringResult` in ``last_citation_result`` for
       ``pipeline.py`` to read via ``hasattr``.  This keeps the pipeline
       decoupled from this class (no ``isinstance`` check required).

    The :class:`~src.generation.base.GeneratorLike` protocol is fully
    satisfied — ``generate()`` has an identical signature to the base
    generator.

    Args:
        base_generator: Any ``GeneratorLike`` (``OpenAICompatibleGenerator``,
            ``AnthropicCompatibleGenerator``, ``ExtractiveGenerator``, etc.).
        scorer: :class:`~src.evaluation.hhem_scorer.NLIScorer` instance.
            When ``None``, citation parsing still runs but HHEM scoring is
            skipped (``scorer_disabled=True`` in the result).
        nli_threshold: HHEM score cutoff for "supports" (default 0.5).
    """

    def __init__(
        self,
        base_generator: object,
        scorer: "NLIScorer | None" = None,
        nli_threshold: float = NLI_THRESHOLD,
    ) -> None:
        self._base = base_generator
        self._scorer = scorer
        self._threshold = nli_threshold
        self.last_citation_result: CitationScoringResult | None = None

        # Inject CITATION_INSTRUCTION into the base generator's system_prompt.
        # OpenAICompatibleGenerator and AnthropicCompatibleGenerator both store
        # system_prompt as a plain public attribute.  ExtractiveGenerator has
        # no such attribute — the try/except handles that gracefully.
        from src.generation.prompts import CITATION_INSTRUCTION

        base_prompt: str = getattr(self._base, "system_prompt", "") or ""
        if CITATION_INSTRUCTION not in base_prompt:
            try:
                self._base.system_prompt = base_prompt + CITATION_INSTRUCTION  # type: ignore[attr-defined]
            except AttributeError:
                logger.debug(
                    "CitationConstrainedGenerator: base generator %s has no "
                    "writable system_prompt — citation instruction not injected.",
                    type(self._base).__name__,
                )

    def generate(self, question: str, contexts: list[Document]) -> GenerationResult:
        """Generate an answer and score citation precision.

        Delegates to the wrapped generator, then parses ``[N]`` citations from
        the result and (optionally) scores them with HHEM.  The citation result
        is stored in ``self.last_citation_result`` for the pipeline to read.

        Args:
            question: The query string.
            contexts: Retrieved passages (1-indexed in the formatted prompt).

        Returns:
            :class:`~src.generation.base.GenerationResult` from the base
            generator — unchanged.
        """
        result: GenerationResult = self._base.generate(question, contexts)  # type: ignore[attr-defined]

        if self._scorer is not None:
            citation_result = score_citation_precision(
                result.text, contexts, self._scorer, self._threshold
            )
        else:
            cited = parse_citation_indices(result.text)
            citation_result = CitationScoringResult(
                citation_count=len(cited),
                citation_precision=None,
                cited_indices=cited,
                hhem_scores={},
                scorer_disabled=True,
            )

        self.last_citation_result = citation_result
        return result
