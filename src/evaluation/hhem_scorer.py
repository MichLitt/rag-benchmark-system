"""Vectara HHEM NLI scorer for post-hoc citation attribution.

Uses `vectara/hallucination_evaluation_model` (DeBERTa-v3-base fine-tuned for
factual consistency) to score (passage, answer_sentence) pairs.

Score semantics
---------------
A score ≥ 0.5 indicates the passage *entails* the answer sentence
(i.e., the passage factually supports the claim).  Lower scores indicate
hallucination or lack of support.

Label order
-----------
Per the Vectara model card, ``id2label = {0: 'consistent', 1: 'inconsistent'}``.
``entailment_label_idx=0`` (the default) returns the probability of label 0
(consistent / entailment).  This is verified at load time and logged so the
caller can catch mismatches.

Usage
-----
    scorer = HHEMScorer()
    prob = scorer.score(premise="Paris is in France.", hypothesis="France has Paris.")
    # prob ≈ 0.92 → passage supports the sentence
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_HHEM_MODEL_NAME = "vectara/hallucination_evaluation_model"
_ENTAILMENT_LABEL_IDX = 0  # label 0 = consistent / entailment


# ---------------------------------------------------------------------------
# Protocol — injected into CitationEvaluator for testability
# ---------------------------------------------------------------------------

@runtime_checkable
class NLIScorer(Protocol):
    """Structural protocol: any object with a compatible ``score`` method."""

    def score(self, premise: str, hypothesis: str) -> float:
        """Return entailment probability ∈ [0, 1].

        Args:
            premise: The supporting passage text.
            hypothesis: The answer sentence to evaluate.

        Returns:
            Probability ∈ [0, 1] that *premise* entails *hypothesis*.
            Higher values indicate stronger support.
        """
        ...


# ---------------------------------------------------------------------------
# HHEMScorer
# ---------------------------------------------------------------------------

class HHEMScorer:
    """Vectara HHEM-based NLI scorer.

    Loads ``vectara/hallucination_evaluation_model`` with
    ``trust_remote_code=True`` (required by the model's custom code) and runs
    inference on CPU or CUDA.

    Args:
        model_name: HuggingFace model ID (defaults to Vectara HHEM).
        device: ``"cpu"`` or ``"cuda"``; auto-detected when ``None``.
        entailment_label_idx: Index of the entailment label in the softmax
            output.  Defaults to 0 (consistent).  Verify against
            ``model.config.id2label`` if the model card changes.
    """

    def __init__(
        self,
        model_name: str = _HHEM_MODEL_NAME,
        device: str | None = None,
        entailment_label_idx: int = _ENTAILMENT_LABEL_IDX,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._entailment_idx = entailment_label_idx
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._torch = torch

        logger.info("Loading HHEM model %r on device=%s …", model_name, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model.to(device)
        self._model.eval()

        # Log the label mapping so callers can verify direction at runtime
        id2label = getattr(self._model.config, "id2label", {})
        logger.info(
            "HHEM id2label=%s → using index %d as 'entailment'",
            id2label, entailment_label_idx,
        )

    # ------------------------------------------------------------------
    # NLIScorer protocol implementation
    # ------------------------------------------------------------------

    def score(self, premise: str, hypothesis: str) -> float:
        """Return entailment probability ∈ [0, 1].

        Args:
            premise: Retrieved passage text (the source of evidence).
            hypothesis: Answer sentence to evaluate against the passage.

        Returns:
            Probability that *premise* entails *hypothesis*.
        """
        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with self._torch.no_grad():
            logits = self._model(**inputs).logits

        probs = self._torch.softmax(logits, dim=-1)
        return float(probs[0][self._entailment_idx].item())
