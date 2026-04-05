"""Quick connectivity check for configured LLM APIs.

Tests each API with a single-turn request and prints the result.
Reads credentials from .env (or environment).

Usage:
    uv run python scripts/check_llm_apis.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(override=True)

QUESTION = "What is the capital of France? Answer in one word."
CONTEXT_TEXT = "France is a country in Western Europe. Its capital city is Paris."


def _doc(text: str, title: str = "France"):
    from src.types import Document
    return Document(doc_id="check_d1", text=text, title=title)


def check_minimax_anthropic() -> bool:
    """MiniMax M2.7 via Anthropic SDK."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    api_base = os.getenv("MINIMAX_ANTHROPIC_BASE_URL", "").strip()

    if not api_key:
        print("[minimax-m2.7] SKIP — ANTHROPIC_API_KEY not set")
        return False
    if not api_base:
        print("[minimax-m2.7] SKIP — MINIMAX_ANTHROPIC_BASE_URL not set")
        return False

    print(f"[minimax-m2.7] Calling {api_base} ...")
    try:
        from src.generation.anthropic_compatible import AnthropicCompatibleGenerator
        model_name = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")
        gen = AnthropicCompatibleGenerator(
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            max_output_tokens=256,
            timeout_sec=60,
        )
        result = gen.generate(QUESTION, [_doc(CONTEXT_TEXT)])
        print(f"[minimax-m2.7] OK — answer: {result.text!r}")
        print(f"             tokens: in={result.input_tokens} out={result.output_tokens}")
        return True
    except Exception as exc:
        print(f"[minimax-m2.7] FAIL — {exc}")
        return False


def check_glm() -> bool:
    """GLM-5 via OpenAI-compatible API."""
    api_key = os.getenv("GLM_API_KEY", "").strip()
    api_base = os.getenv("GLM_BASE_URL", "").strip()

    if not api_key:
        print("[glm-5] SKIP — GLM_API_KEY not set")
        return False
    if not api_base:
        print("[glm-5] SKIP — GLM_BASE_URL not set")
        return False

    print(f"[glm-5] Calling {api_base} ...")
    try:
        from src.generation.openai_compatible import OpenAICompatibleGenerator
        model_name = os.getenv("GLM_MODEL", "glm-5.1")
        gen = OpenAICompatibleGenerator(
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            max_output_tokens=256,
            max_completion_tokens=2048,
            timeout_sec=60,
            reasoning_split="auto",
        )
        result = gen.generate(QUESTION, [_doc(CONTEXT_TEXT)])
        print(f"[glm-5] OK — answer: {result.text!r}")
        print(f"        tokens: in={result.input_tokens} out={result.output_tokens}")
        return True
    except Exception as exc:
        print(f"[glm-5] FAIL — {exc}")
        return False


def main() -> None:
    print("=" * 60)
    print("LLM API connectivity check")
    print("=" * 60)

    results = {
        "minimax-m2.7 (Anthropic SDK)": check_minimax_anthropic(),
        "glm (OpenAI-compat)": check_glm(),
    }

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL/SKIP"
        print(f"  {status:10s} {name}")

    if not any(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
