from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.query.base import QueryExpanderLike
from src.query.decomposition import (
    DEFAULT_DECOMPOSITION_SYSTEM_PROMPT,
    DEFAULT_DECOMPOSITION_USER_PROMPT_TEMPLATE,
    HotpotDecomposeExpander,
)
from src.query.hyde import (
    DEFAULT_HYDE_SYSTEM_PROMPT,
    DEFAULT_HYDE_USER_PROMPT_TEMPLATE,
    HOTPOT_HYDE_SYSTEM_PROMPT,
    HOTPOT_HYDE_USER_PROMPT_TEMPLATE,
    HyDEExpander,
)


def _resolve_setting(
    primary_cfg: dict[str, Any],
    fallback_cfg: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    value = primary_cfg.get(key)
    if value not in (None, ""):
        return value
    fallback_value = fallback_cfg.get(key)
    if fallback_value not in (None, ""):
        return fallback_value
    return default


def _normalize_dataset_name(dataset_name: str) -> str:
    return dataset_name.strip().lower()


def _resolve_query_cache_dir(
    query_cfg: dict[str, Any],
    dataset_name: str | None,
) -> Path | None:
    raw_cache_dir = query_cfg.get("cache_dir")
    if raw_cache_dir in (None, ""):
        raw_cache_dir = Path("experiments") / "cache" / "query_expansion"
    cache_dir = Path(raw_cache_dir)
    dataset_key = _normalize_dataset_name(dataset_name or "")
    if dataset_key:
        return cache_dir / dataset_key
    return cache_dir


def _parse_dataset_allowlist(value: Any) -> set[str]:
    if value in (None, "", []):
        return set()
    if isinstance(value, str):
        return {_normalize_dataset_name(token) for token in value.split(",") if token.strip()}
    if isinstance(value, list):
        return {_normalize_dataset_name(str(token)) for token in value if str(token).strip()}
    raise ValueError(f"Unsupported query_expansion.datasets value: {value!r}")


def resolve_query_expansion_mode(cfg: dict[str, Any], dataset_name: str | None = None) -> str:
    query_cfg = cfg.get("query_expansion", {})
    configured_mode = str(query_cfg.get("mode", "off")).strip().lower()
    if configured_mode not in {"off", "auto", "hyde", "hotpot_hyde", "hotpot_decompose"}:
        raise ValueError(f"Unsupported query_expansion.mode: {configured_mode}")

    dataset_key = _normalize_dataset_name(dataset_name or "")
    allowed_datasets = _parse_dataset_allowlist(query_cfg.get("datasets"))
    if dataset_key and allowed_datasets and dataset_key not in allowed_datasets:
        return "off"

    if configured_mode == "auto":
        if dataset_key == "nq":
            return "hyde"
        return "off"
    return configured_mode


def build_query_expander(
    cfg: dict[str, Any],
    *,
    dataset_name: str | None = None,
    mode_override: str | None = None,
) -> QueryExpanderLike | None:
    load_dotenv()
    generation_cfg = cfg.get("generation", {})
    query_cfg = cfg.get("query_expansion", {})
    mode = str(mode_override or resolve_query_expansion_mode(cfg, dataset_name)).strip().lower()
    if mode == "off":
        return None
    if mode not in {"hyde", "hotpot_hyde", "hotpot_decompose"}:
        raise ValueError(f"Unsupported query_expansion.mode: {mode}")

    model = str(_resolve_setting(query_cfg, generation_cfg, "model", "")).strip()
    if not model:
        raise ValueError(
            "query_expansion.model must be set, or generation.model must be available for HyDE."
        )

    api_key_env = str(_resolve_setting(query_cfg, generation_cfg, "api_key_env", "LLM_API_KEY")).strip()
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise ValueError(
            "Environment variable "
            f"{api_key_env} is required for query expansion when query_expansion.mode={mode}."
        )

    api_base = str(query_cfg.get("api_base", "")).strip()
    api_base_env = str(_resolve_setting(query_cfg, generation_cfg, "api_base_env", "LLM_BASE_URL")).strip()
    if not api_base and api_base_env:
        api_base = os.getenv(api_base_env, "").strip()
    if not api_base:
        raise ValueError(
            "Query expansion base URL is required. "
            f"Set query_expansion.api_base or environment variable {api_base_env}."
        )

    system_prompt = str(query_cfg.get("system_prompt", "")).strip()
    user_prompt_template = str(query_cfg.get("user_prompt_template", "")).strip()
    if not system_prompt:
        if mode == "hotpot_hyde":
            system_prompt = HOTPOT_HYDE_SYSTEM_PROMPT
        elif mode == "hotpot_decompose":
            system_prompt = DEFAULT_DECOMPOSITION_SYSTEM_PROMPT
        else:
            system_prompt = DEFAULT_HYDE_SYSTEM_PROMPT
    if not user_prompt_template:
        if mode == "hotpot_hyde":
            user_prompt_template = HOTPOT_HYDE_USER_PROMPT_TEMPLATE
        elif mode == "hotpot_decompose":
            user_prompt_template = DEFAULT_DECOMPOSITION_USER_PROMPT_TEMPLATE
        else:
            user_prompt_template = DEFAULT_HYDE_USER_PROMPT_TEMPLATE

    if mode == "hotpot_decompose":
        max_completion_tokens = int(query_cfg.get("max_completion_tokens", 512))
        return HotpotDecomposeExpander(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=float(query_cfg.get("temperature", 0.0)),
            max_completion_tokens=max_completion_tokens,
            timeout_sec=int(_resolve_setting(query_cfg, generation_cfg, "timeout_sec", 60)),
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            include_original_query=bool(query_cfg.get("include_original_query", True)),
            max_queries=int(query_cfg.get("max_queries", 3)),
            cache_dir=_resolve_query_cache_dir(query_cfg, dataset_name),
        )

    return HyDEExpander(
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=float(query_cfg.get("temperature", 0.0)),
        max_completion_tokens=int(query_cfg.get("max_completion_tokens", 256)),
        timeout_sec=int(_resolve_setting(query_cfg, generation_cfg, "timeout_sec", 60)),
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
    )
