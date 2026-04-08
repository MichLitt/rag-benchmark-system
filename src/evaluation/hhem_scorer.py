from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HHEMResult:
    score: float
    """Factual consistency score in [0, 1]. Higher = more consistent (less hallucinated)."""
    is_consistent: bool
    """True if score >= threshold."""
    error: str = ""
    """Non-empty if scoring failed."""


class HHEMScorer:
    """NLI-based factual consistency scorer using Vectara HHEM.

    Uses vectara/hallucination_evaluation_model (a fine-tuned DeBERTa
    cross-encoder) to score whether a generated answer is supported by a
    source passage.

    Reference: https://huggingface.co/vectara/hallucination_evaluation_model

    Usage:
        scorer = HHEMScorer()
        result = scorer.score(
            source="The sky is blue.",
            summary="The sky has a blue color.",
        )
        print(result.score, result.is_consistent)
    """

    MODEL_NAME = "vectara/hallucination_evaluation_model"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        # Lazy imports — transformers/torch are optional dependencies.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self._threshold = threshold
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model.eval()
        self._model.to(self._device)

    def score(self, source: str, summary: str) -> HHEMResult:
        """Score whether `summary` is factually consistent with `source`.

        HHEM outputs a binary classification (0=hallucinated, 1=consistent).
        We return the softmax probability of label=1 as the score.

        Args:
            source: The reference passage (ground truth / retrieved context).
            summary: The generated answer or claim to evaluate.

        Returns:
            HHEMResult with score in [0, 1].
        """
        try:
            import torch

            inputs = self._tokenizer(
                source,
                summary,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits

            probs = torch.softmax(logits, dim=-1)
            # Label ordering: verify with model.config.id2label if behaviour seems wrong.
            # HHEM model card states: label 0 = hallucinated, label 1 = factually consistent.
            score = float(probs[0][1].item())
            return HHEMResult(score=score, is_consistent=score >= self._threshold)

        except Exception as exc:
            return HHEMResult(score=0.0, is_consistent=False, error=str(exc))

    def score_batch(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[HHEMResult]:
        """Score a list of (source, summary) pairs sequentially.

        Args:
            pairs: List of (source, summary) tuples.

        Returns:
            List of HHEMResult, one per pair, in the same order.
        """
        return [self.score(source, summary) for source, summary in pairs]
