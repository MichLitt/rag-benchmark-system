"""Answer post-processing for generation outputs.

Strips hedging language from model-generated answers before metric computation.
Models frequently output preambles like "According to the context, the answer is Paris"
which cause EM failures even when the extracted fact is correct.

The HotpotQA dataset-specific prompt instructs the model to output
"Final answer: <answer>" — this prefix is also handled here.
"""
from __future__ import annotations

import re

# Patterns applied in order; first match wins (no double-stripping).
# Most-specific patterns come before general ones to avoid over-stripping.
_HEDGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^according to (?:the )?(?:context|passage|text|document|retrieved (?:passage|context|document)s?)[,.]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^based on (?:the )?(?:context|passage|text|document|retrieved (?:passage|context|document)s?)[,.]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^the (?:context|passage|text|document) (?:mentions?|states?|says?|indicates?|shows?) that\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^as (?:mentioned|stated|indicated) in (?:the )?(?:context|passage|text)[,.]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^the answer (?:is|to this question is)\s*[:\-]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^from (?:the )?(?:context|passage|retrieved (?:passage|context)s?)[,.]?\s*",
        re.IGNORECASE,
    ),
    # HotpotQA dataset-specific prompt instructs model to output "Final answer: X"
    re.compile(
        r"^final answer\s*[:\-]\s*",
        re.IGNORECASE,
    ),
]

# After stripping a hedge prefix, strip any leading punctuation artifacts.
_LEADING_PUNCT: re.Pattern[str] = re.compile(r"^[,\-:;\s]+")


def strip_hedging(text: str) -> tuple[str, bool]:
    """Remove a hedging prefix from a model-generated answer.

    Returns ``(cleaned_text, was_hedged)``. ``was_hedged`` is ``True`` if any
    pattern matched, allowing downstream callers to track hedge detection rate.

    Only the first matching pattern is applied (no recursive stripping).
    If stripping would produce an empty string the original text is returned
    unchanged with ``was_hedged=True`` so callers can distinguish the case
    from a genuinely short answer.

    Args:
        text: Raw answer text from the generator.

    Returns:
        Tuple of ``(answer, was_hedged)`` where ``answer`` is the (possibly
        unchanged) text and ``was_hedged`` indicates whether a pattern fired.
    """
    stripped = text.strip()
    for pattern in _HEDGE_PATTERNS:
        m = pattern.match(stripped)
        if m:
            remainder = _LEADING_PUNCT.sub("", stripped[m.end():]).strip()
            if remainder:
                return remainder, True
            # Stripping left nothing — preserve original to avoid empty answers.
            return text.strip(), True
    return text.strip(), False


def postprocess_answer(text: str) -> tuple[str, bool]:
    """Entry point called by the pipeline for answer post-processing.

    A thin wrapper around :func:`strip_hedging` that provides a stable
    public interface. Additional normalization steps (e.g. trailing citation
    removal) can be added here without touching ``pipeline.py``.

    Args:
        text: Raw generated answer.

    Returns:
        Tuple ``(cleaned_text, was_hedged)``.
    """
    return strip_hedging(text)
