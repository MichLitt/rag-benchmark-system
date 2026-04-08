from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from src.generation.anthropic_compatible import AnthropicCompatibleGenerator
from src.generation.base import GeneratorLike
from src.generation.extractive import ExtractiveGenerator
from src.generation.openai_compatible import DEFAULT_SYSTEM_PROMPT, OpenAICompatibleGenerator


def build_generator(cfg: dict[str, Any]) -> GeneratorLike:
    load_dotenv()
    generation_cfg = cfg.get("generation", {})
    mode = str(generation_cfg.get("mode", "extractive")).strip().lower()

    if mode == "extractive":
        return ExtractiveGenerator()

    if mode in {"anthropic", "anthropic_compatible"}:
        api_key_env = str(generation_cfg.get("api_key_env", "ANTHROPIC_API_KEY")).strip()
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env} is required for Anthropic generation."
            )
        api_base = str(generation_cfg.get("api_base", "")).strip()
        api_base_env = str(generation_cfg.get("api_base_env", "")).strip()
        if not api_base and api_base_env:
            api_base = os.getenv(api_base_env, "").strip()
        model = str(generation_cfg.get("model", "")).strip()
        if not model:
            raise ValueError("generation.model must be set when generation.mode uses an LLM.")
        max_output_tokens = int(generation_cfg.get("max_output_tokens", 128))
        generator: GeneratorLike = AnthropicCompatibleGenerator(
            model=model,
            api_key=api_key,
            api_base=api_base or None,
            system_prompt=str(generation_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
            temperature=float(generation_cfg.get("temperature", 0.0)),
            max_output_tokens=max_output_tokens,
            timeout_sec=int(generation_cfg.get("timeout_sec", 60)),
            input_price_per_1m=float(generation_cfg.get("input_price_per_1m", 0.0)),
            output_price_per_1m=float(generation_cfg.get("output_price_per_1m", 0.0)),
        )
    elif mode not in {"llm", "openai_compatible"}:
        raise ValueError(f"Unsupported generation mode: {mode}")
    else:
        model = str(generation_cfg.get("model", "")).strip()
        if not model:
            raise ValueError("generation.model must be set when generation.mode uses an LLM.")

        api_key_env = str(generation_cfg.get("api_key_env", "OPENAI_API_KEY")).strip()
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is required for LLM generation.")

        api_base = str(generation_cfg.get("api_base", "")).strip()
        api_base_env = str(generation_cfg.get("api_base_env", "LLM_BASE_URL")).strip()
        if not api_base and api_base_env:
            api_base = os.getenv(api_base_env, "").strip()
        if not api_base:
            raise ValueError(
                "LLM base URL is required for LLM generation. "
                f"Set generation.api_base or environment variable {api_base_env}."
            )

        max_output_tokens = int(generation_cfg.get("max_output_tokens", 128))
        max_completion_tokens_raw = generation_cfg.get("max_completion_tokens")
        max_completion_tokens = (
            int(max_completion_tokens_raw)
            if max_completion_tokens_raw not in (None, "")
            else None
        )
        if model.strip().lower().startswith("minimax") and max_completion_tokens is None:
            max_completion_tokens = max(max_output_tokens, 512)

        generator = OpenAICompatibleGenerator(
            model=model,
            api_key=api_key,
            api_base=api_base,
            system_prompt=str(generation_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
            temperature=float(generation_cfg.get("temperature", 0.0)),
            max_output_tokens=max_output_tokens,
            max_completion_tokens=max_completion_tokens,
            reasoning_split=generation_cfg.get("reasoning_split", "auto"),
            timeout_sec=int(generation_cfg.get("timeout_sec", 60)),
            input_price_per_1m=float(generation_cfg.get("input_price_per_1m", 0.0)),
            output_price_per_1m=float(generation_cfg.get("output_price_per_1m", 0.0)),
        )

    # --- Phase 5: dataset-specific prompt routing ---
    # Applied when system_prompt is empty/unset in config AND dataset is specified.
    # Priority: explicit system_prompt in config > dataset routing > DEFAULT_SYSTEM_PROMPT
    # (resolve_system_prompt handles this priority internally).
    dataset = str(generation_cfg.get("dataset", "")).strip().lower()
    explicit_prompt = str(generation_cfg.get("system_prompt", "")).strip()
    if dataset and not explicit_prompt:
        from src.generation.prompts import resolve_system_prompt
        resolved = resolve_system_prompt(dataset=dataset)
        if hasattr(generator, "system_prompt"):
            generator.system_prompt = resolved  # type: ignore[union-attr]

    # --- Phase 5: citation-constrained wrapping ---
    # Wraps the base generator in CitationConstrainedGenerator when enabled.
    # Loads HHEMScorer once here (amortised across all N samples in the eval loop).
    citation_constrained = bool(generation_cfg.get("citation_constrained", False))
    if citation_constrained:
        from src.generation.citation_wrapper import CitationConstrainedGenerator
        score_citations = bool(generation_cfg.get("score_citations", True))
        nli_threshold = float(generation_cfg.get("nli_threshold", 0.5))
        scorer = None
        if score_citations:
            from src.evaluation.hhem_scorer import HHEMScorer
            # HHEM loaded once at construction — DeBERTa inference is amortised
            # across all queries in the evaluation run, not re-loaded per sample.
            scorer = HHEMScorer()
        generator = CitationConstrainedGenerator(
            base_generator=generator,
            scorer=scorer,
            nli_threshold=nli_threshold,
        )

    return generator
