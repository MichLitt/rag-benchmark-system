from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.generation.openai_compatible import (
    _extract_text,
    _is_minimax_model,
    _strip_think_blocks,
)


DEFAULT_DECOMPOSITION_SYSTEM_PROMPT = (
    "Rewrite the question into search queries for multi-hop retrieval. "
    "Return one JSON object only in the exact format {\"queries\": [\"...\", \"...\"]}. "
    "Each item in queries must be a search query, not an explanation or answer sentence. "
    "Do not output markdown, notes, bullet points, or reasoning. "
    "Do not output <think> tags. "
    "If unsure, still return 2 concise search queries."
)
DEFAULT_DECOMPOSITION_USER_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Return 2 or 3 concise search queries.\n"
    "The queries should target bridge entities, answer-side entities, or relation clues.\n"
    "Queries may contain person names, film titles, organizations, places, or key relations.\n"
    "Prefer short keyword-style search queries, not explanations.\n"
    "Good examples:\n"
    "1. {{\"queries\": [\"Tim Burton Ed Wood cast\", \"Ed Wood lead actor\"]}}\n"
    "2. {{\"queries\": [\"Shirley Temple government position\", \"Kiss and Tell 1945 Corliss Archer\"]}}\n"
    "Bad examples:\n"
    "1. {{\"queries\": [\"The answer is Johnny Depp\", \"I would search for the actor in Ed Wood\"]}}\n"
    "2. Here are some queries:\n- Tim Burton Ed Wood\n- Johnny Depp\n"
    "Return JSON only."
)

HIGH_CONFIDENCE_META_PATTERNS = (
    r"^(?:here(?: are|'re)?|below are|these are)\s+(?:the\s+)?queries\b",
    r"^(?:i would|i'd|we should|we need to)\b",
    r"^(?:the answer is|answer:)\b",
    r"^(?:return|output)\s+(?:json|queries?)\b",
)
TRAILING_PUNCTUATION = " \t\r\n.,!?;:"


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fenced_match = re.fullmatch(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()
    return text


def _unwrap_think_tags(text: str) -> str:
    return re.sub(r"</?think>", "\n", text, flags=re.IGNORECASE)


class HotpotDecomposeExpander:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        api_base: str,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
        timeout_sec: int = 60,
        system_prompt: str = DEFAULT_DECOMPOSITION_SYSTEM_PROMPT,
        user_prompt_template: str = DEFAULT_DECOMPOSITION_USER_PROMPT_TEMPLATE,
        include_original_query: bool = True,
        max_queries: int = 3,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model = model.strip()
        if not self.model:
            raise ValueError("Decomposition model must be non-empty.")
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("Decomposition api_key must be non-empty.")
        self.api_base = api_base.rstrip("/")
        if not self.api_base:
            raise ValueError("Decomposition api_base must be non-empty.")
        self.temperature = float(temperature)
        self.max_completion_tokens = int(max_completion_tokens)
        if self.max_completion_tokens <= 0:
            raise ValueError(
                "Decomposition max_completion_tokens must be > 0, "
                f"got {self.max_completion_tokens}"
            )
        self.timeout_sec = int(timeout_sec)
        self.system_prompt = system_prompt.strip() or DEFAULT_DECOMPOSITION_SYSTEM_PROMPT
        self.user_prompt_template = (
            user_prompt_template.strip() or DEFAULT_DECOMPOSITION_USER_PROMPT_TEMPLATE
        )
        self.include_original_query = bool(include_original_query)
        self.max_queries = int(max_queries)
        if self.max_queries <= 0:
            raise ValueError(f"max_queries must be > 0, got {self.max_queries}")
        self.cache_dir = Path(cache_dir) if cache_dir not in (None, "") else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_expansion_metadata: dict[str, Any] = {}

    def _request_token_budget(self) -> tuple[str, int]:
        if _is_minimax_model(self.model):
            return "max_completion_tokens", self.max_completion_tokens
        return "max_tokens", self.max_completion_tokens

    def _build_request_body(self, question: str) -> dict[str, Any]:
        budget_key, budget_value = self._request_token_budget()
        user_prompt = self.user_prompt_template.format(question=question.strip())
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "reasoning_split": False,
        }
        body[budget_key] = budget_value
        return body

    def _cache_key(self, question: str) -> str:
        payload = json.dumps(
            {
                "question": question.strip(),
                "model": self.model,
                "system_prompt": self.system_prompt,
                "user_prompt_template": self.user_prompt_template,
                "include_original_query": self.include_original_query,
                "max_queries": self.max_queries,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, question: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self._cache_key(question)}.json"

    def _failure_cache_path(self, question: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / "failures" / f"{self._cache_key(question)}.json"

    def _set_last_metadata(
        self,
        *,
        question: str,
        cache_hit: bool,
        failure_reason: str,
        used_fallback: bool,
        raw_content: str = "",
        raw_text: str = "",
        stripped_text: str = "",
        raw_queries: list[str] | None = None,
        normalized_queries: list[str] | None = None,
        salvage_stage: str = "",
    ) -> None:
        self.last_expansion_metadata = {
            "question": question.strip(),
            "cache_key": self._cache_key(question),
            "cache_hit": bool(cache_hit),
            "failure_reason": failure_reason.strip(),
            "used_fallback": bool(used_fallback),
            "raw_content": raw_content,
            "raw_text": raw_text,
            "stripped_text": stripped_text,
            "raw_queries": list(raw_queries or []),
            "normalized_queries": list(normalized_queries or []),
            "salvage_stage": salvage_stage,
        }

    def get_last_expansion_metadata(self) -> dict[str, Any]:
        return dict(self.last_expansion_metadata)

    def _write_failure_diagnostics(self) -> None:
        failure_reason = str(self.last_expansion_metadata.get("failure_reason", "")).strip()
        if not failure_reason:
            return
        failure_path = self._failure_cache_path(str(self.last_expansion_metadata.get("question", "")))
        if failure_path is None:
            return
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(self.last_expansion_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_cached_queries(self, question: str) -> list[str] | None:
        cache_path = self._cache_path(question)
        if cache_path is None or not cache_path.exists():
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        raw_queries = payload.get("queries")
        if not isinstance(raw_queries, list):
            return None
        queries = [str(item).strip() for item in raw_queries if str(item).strip()]
        if queries:
            self._set_last_metadata(
                question=question,
                cache_hit=True,
                failure_reason="",
                used_fallback=False,
                normalized_queries=queries,
                salvage_stage="cache",
            )
        return queries or None

    def _save_cached_queries(self, question: str, queries: list[str]) -> None:
        cache_path = self._cache_path(question)
        if cache_path is None:
            return
        cache_path.write_text(
            json.dumps(
                {
                    "question": question.strip(),
                    "queries": queries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _extract_queries_from_text(self, text: str) -> list[str]:
        stripped_text = _strip_code_fences(text).strip()
        candidates: list[str] = []
        try:
            payload = json.loads(stripped_text)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            raw_queries = payload.get("queries")
            if isinstance(raw_queries, list):
                candidates.extend(str(item).strip() for item in raw_queries)
        elif isinstance(payload, list):
            candidates.extend(str(item).strip() for item in payload)

        if candidates:
            return candidates

        for line in stripped_text.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                candidates.append(cleaned)
        if len(candidates) > 1:
            return candidates
        if len(candidates) == 1 and " - " not in candidates[0]:
            return candidates

        bullet_parts = [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part).strip()
            for part in re.split(r"\s+-\s+", stripped_text)
            if part.strip()
        ]
        if len(bullet_parts) > 1:
            return bullet_parts

        return []

    def _extract_quoted_queries(self, text: str) -> list[str]:
        quoted: list[str] = []
        for match in re.findall(r'"([^"\n]{3,400})"', text):
            candidate = " ".join(match.split()).strip()
            if candidate:
                quoted.append(candidate)
        return quoted

    def _sentence_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []
        for part in re.split(r"[\n]|(?<=[.!?])\s+", text):
            cleaned = " ".join(part.split()).strip()
            if cleaned:
                candidates.append(cleaned)
        return candidates

    def _truncate_query_words(self, text: str, max_words: int = 12) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]).strip()

    def _looks_like_meta(self, text: str) -> bool:
        lowered = text.strip().lower()
        return any(re.search(pattern, lowered) for pattern in HIGH_CONFIDENCE_META_PATTERNS)

    def _normalize_candidate(self, item: str) -> tuple[str, str | None]:
        query = " ".join(item.split()).strip().strip("`").strip()
        if query.startswith('"') and query.endswith('"') and len(query) >= 2:
            query = query[1:-1].strip()
        query = query.strip(TRAILING_PUNCTUATION).strip()
        if not query:
            return "", "empty"
        if query.startswith("{") or query.endswith("}"):
            return "", "json_artifact"
        if self._looks_like_meta(query):
            return "", "meta_only"
        if query.lower() in {"i do not know", "i do not know."}:
            return "", "meta_only"
        return query, None

    def _failure_reason_from_filters(self, raw_queries: list[str]) -> str:
        if not raw_queries:
            return "empty_content"
        normalized_candidates = [self._normalize_candidate(item) for item in raw_queries]
        filter_reasons = [reason for _, reason in normalized_candidates if reason]
        if filter_reasons and all(reason == "meta_only" for reason in filter_reasons):
            return "meta_only"
        if all(len(" ".join(item.split()).strip().split()) > 12 for item in raw_queries if item.strip()):
            return "over_length_only"
        return "all_candidates_filtered"

    def _normalize_queries(self, question: str, raw_queries: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        truncated_candidates: list[str] = []
        for item in raw_queries:
            query, _ = self._normalize_candidate(item)
            if not query:
                continue
            if len(query.split()) > 12:
                truncated = self._truncate_query_words(query, max_words=12)
                if truncated:
                    truncated_candidates.append(truncated)
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(query)

        if not normalized and truncated_candidates:
            for query in truncated_candidates:
                key = query.lower()
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(query)

        if self.include_original_query:
            original = " ".join(question.split()).strip()
            original_key = original.lower()
            if original and original_key not in seen:
                normalized.insert(0, original)

        return normalized[: self.max_queries]

    def _extract_queries_with_stage(self, text: str) -> tuple[list[str], str]:
        stripped = _strip_code_fences(text).strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            raw_queries = payload.get("queries")
            if isinstance(raw_queries, list):
                return [str(item).strip() for item in raw_queries], "strict_json"
        elif isinstance(payload, list):
            return [str(item).strip() for item in payload], "strict_json"

        if text.strip().startswith("```"):
            fenced_candidates = self._extract_queries_from_text(text)
            if fenced_candidates:
                return fenced_candidates, "fenced_json"

        quoted_candidates = self._extract_quoted_queries(text)
        if quoted_candidates:
            return quoted_candidates, "quoted"

        line_candidates = self._extract_queries_from_text(text)
        if line_candidates:
            return line_candidates, "line_based"

        sentence_candidates = self._sentence_candidates(text)
        if sentence_candidates:
            return sentence_candidates, "sentence_split"

        return [], ""

    def _fallback_queries(self, question: str) -> list[str]:
        return [" ".join(question.split()).strip()]

    def expand_queries(self, question: str) -> list[str]:
        cached = self._load_cached_queries(question)
        if cached is not None:
            return cached

        body = self._build_request_body(question)
        request = urllib.request.Request(
            url=f"{self.api_base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._set_last_metadata(
                question=question,
                cache_hit=False,
                failure_reason="http_error",
                used_fallback=False,
                raw_content=detail,
            )
            self._write_failure_diagnostics()
            raise RuntimeError(f"Decomposition request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            self._set_last_metadata(
                question=question,
                cache_hit=False,
                failure_reason="url_error",
                used_fallback=False,
            )
            self._write_failure_diagnostics()
            raise RuntimeError(f"Decomposition request failed: {exc.reason}") from exc

        choices = payload.get("choices", [])
        if not choices:
            self._set_last_metadata(
                question=question,
                cache_hit=False,
                failure_reason="empty_content",
                used_fallback=False,
            )
            self._write_failure_diagnostics()
            raise RuntimeError("Decomposition response did not contain any choices.")

        message = choices[0].get("message", {})
        raw_content = _extract_text(message.get("content"))
        raw_text = _unwrap_think_tags(raw_content)
        stripped_text = _strip_think_blocks(raw_content)
        raw_queries, salvage_stage = self._extract_queries_with_stage(stripped_text)
        if not raw_queries:
            raw_queries, salvage_stage = self._extract_queries_with_stage(raw_text)
        queries = self._normalize_queries(question, raw_queries)

        failure_reason = ""
        if not raw_content.strip():
            failure_reason = "empty_content"
        elif not raw_queries:
            failure_reason = "json_parse_failed"
        elif not queries:
            failure_reason = self._failure_reason_from_filters(raw_queries)

        if not queries:
            fallback_queries = self._fallback_queries(question)
            self._set_last_metadata(
                question=question,
                cache_hit=False,
                failure_reason=failure_reason or "all_candidates_filtered",
                used_fallback=True,
                raw_content=raw_content,
                raw_text=raw_text,
                stripped_text=stripped_text,
                raw_queries=raw_queries,
                normalized_queries=fallback_queries,
                salvage_stage=salvage_stage,
            )
            self._write_failure_diagnostics()
            return fallback_queries

        self._set_last_metadata(
            question=question,
            cache_hit=False,
            failure_reason="",
            used_fallback=False,
            raw_content=raw_content,
            raw_text=raw_text,
            stripped_text=stripped_text,
            raw_queries=raw_queries,
            normalized_queries=queries,
            salvage_stage=salvage_stage,
        )
        self._save_cached_queries(question, queries)
        return queries

    def expand(self, question: str) -> str:
        return " || ".join(self.expand_queries(question))
