"""LLM-as-judge faithfulness scoring.

Scores whether a generated answer is supported by the retrieved context.
Uses the same OpenAI-compatible API as the generation backend.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating whether an answer is faithfully "
    "supported by the provided context. You must output ONLY a JSON object."
)

JUDGE_USER_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer: {answer}

Judge whether the answer is supported by the context above.
Output a JSON object with exactly two fields:
- "faithfulness": a float from 0.0 (completely unsupported / hallucinated) to 1.0 (fully supported by context)
- "reasoning": a brief one-sentence explanation

Output ONLY the JSON object, nothing else."""


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float
    reasoning: str
    raw_response: str
    error: str = ""


def _strip_think_blocks(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think>.*", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</think>", " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split()).strip()


def _parse_score(text: str) -> tuple[float, str]:
    """Extract faithfulness score and reasoning from LLM JSON response."""
    text = _strip_think_blocks(text).strip()
    # Try to find JSON in the response
    for pattern in [
        r"\{[^{}]*\}",  # simple JSON object
    ]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
                score = float(obj.get("faithfulness", 0.0))
                score = max(0.0, min(1.0, score))
                reasoning = str(obj.get("reasoning", ""))
                return score, reasoning
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return 0.0, "parse_failure"


def score_faithfulness(
    question: str,
    answer: str,
    context_texts: list[str],
    *,
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    model: str = "MiniMax-M2.5",
    max_completion_tokens: int = 512,
    timeout_sec: int = 30,
) -> FaithfulnessResult:
    """Score faithfulness of an answer given the retrieved context."""
    if not answer.strip() or not context_texts:
        return FaithfulnessResult(score=0.0, reasoning="empty_input", raw_response="")

    context_block = "\n\n".join(
        f"[{i}] {text}" for i, text in enumerate(context_texts, 1)
    )
    user_prompt = JUDGE_USER_TEMPLATE.format(
        context=context_block,
        question=question.strip(),
        answer=answer.strip(),
    )

    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_completion_tokens": max_completion_tokens,
    }

    request = urllib.request.Request(
        url=f"{api_base.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return FaithfulnessResult(
            score=0.0,
            reasoning="api_error",
            raw_response="",
            error=str(exc),
        )

    choices = payload.get("choices", [])
    if not choices:
        return FaithfulnessResult(
            score=0.0, reasoning="no_choices", raw_response="", error="Empty choices"
        )

    raw_text = choices[0].get("message", {}).get("content", "")
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    score, reasoning = _parse_score(raw_text)
    return FaithfulnessResult(
        score=score,
        reasoning=reasoning,
        raw_response=raw_text,
    )
