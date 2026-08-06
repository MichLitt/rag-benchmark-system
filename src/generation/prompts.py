"""Dataset-specific system prompts for generation.

Centralises all system prompt variants so that ``factory.py`` acts only as a
router with no hardcoded strings.

Prompt design rationale
-----------------------
DEFAULT_SYSTEM_PROMPT
    Generic fallback — same wording that was the sole prompt in Phases 1–4.

HOTPOTQA_SYSTEM_PROMPT
    HotpotQA requires two-hop reasoning: the model must bridge two passages to
    reach the answer.  We explicitly ask for step-by-step evidence before the
    final answer, and cap it at ≤5 words so that EM scoring succeeds.
    The "Final answer:" format is intentional — ``postprocess.py`` strips this
    prefix before metric computation.

TRIVIAQA_SYSTEM_PROMPT
    Trivia answers are typically a single name, date, or number.  An aggressive
    ≤3-word cap forces the model to suppress verbosity that tanks EM.

NQ_SYSTEM_PROMPT
    Natural Questions answers are short factual phrases.  ≤10 words is generous
    enough to cover most answers while still suppressing preamble.

CITATION_INSTRUCTION
    Appended to any prompt when ``citation_constrained=true``.  Instructs the
    model to cite passage indices so that ``citation_wrapper.py`` can parse and
    score them with HHEM.
"""
from __future__ import annotations

DEFAULT_SYSTEM_PROMPT: str = (
    "You answer questions based on the retrieved context. "
    "Extract the most relevant information from the context to answer the question. "
    "Keep the answer short and factual — a few words or one sentence. "
    "Do not reveal reasoning. Do not output <think> tags or chain-of-thought. "
    "Output only the final answer."
)

HOTPOTQA_SYSTEM_PROMPT: str = (
    "You answer multi-hop questions that require reasoning across multiple passages. "
    "Read all retrieved passages carefully. "
    "Reason step by step using evidence from the passages, then state the final answer. "
    "The final answer must be ≤5 words: a name, date, yes/no, or short phrase. "
    "Do not output <think> tags. "
    "Output your reasoning followed by 'Final answer: <answer>'."
)

TRIVIAQA_SYSTEM_PROMPT: str = (
    "You answer trivia questions using only the retrieved passages. "
    "The answer is always a single name, date, number, or very short phrase (≤3 words). "
    "Do not explain. Do not output <think> tags. "
    "Output only the answer."
)

NQ_SYSTEM_PROMPT: str = (
    "You answer natural questions using only the retrieved context. "
    "Keep the answer to ≤10 words — a name, entity, or short factual phrase. "
    "Do not explain. Do not output <think> tags. "
    "Output only the answer."
)

# Registry used by factory.py and resolve_system_prompt() for dataset lookup.
DATASET_PROMPTS: dict[str, str] = {
    "hotpotqa": HOTPOTQA_SYSTEM_PROMPT,
    "triviaqa": TRIVIAQA_SYSTEM_PROMPT,
    "nq": NQ_SYSTEM_PROMPT,
}

CITATION_INSTRUCTION: str = (
    "\n\nIMPORTANT: You must cite passage numbers inline in your answer. "
    "Use square-bracket citations like [1] or [2, 3] immediately after each claim. "
    "Example: 'Paris is the capital of France [1]. It has 2 million residents [2, 3].'"
)


def resolve_system_prompt(
    dataset: str | None = None,
    base_prompt: str | None = None,
    add_citation_instruction: bool = False,
) -> str:
    """Return the resolved system prompt for a given context.

    Priority order:
    1. Explicit ``base_prompt`` (non-empty string) — used as-is.
    2. Dataset registry lookup (``DATASET_PROMPTS``).
    3. ``DEFAULT_SYSTEM_PROMPT`` fallback.

    ``CITATION_INSTRUCTION`` is appended last when ``add_citation_instruction``
    is ``True``, regardless of which branch provided the base prompt.

    Args:
        dataset: Dataset name (case-insensitive). Recognised values:
            ``"hotpotqa"``, ``"triviaqa"``, ``"nq"``.
        base_prompt: Explicit system prompt string. When non-empty, takes
            priority over dataset routing.
        add_citation_instruction: Append :data:`CITATION_INSTRUCTION` to the
            resolved prompt.

    Returns:
        Resolved system prompt string.
    """
    if base_prompt and base_prompt.strip():
        prompt = base_prompt.strip()
    elif dataset and dataset.strip().lower() in DATASET_PROMPTS:
        prompt = DATASET_PROMPTS[dataset.strip().lower()]
    else:
        prompt = DEFAULT_SYSTEM_PROMPT

    if add_citation_instruction:
        prompt = prompt + CITATION_INSTRUCTION

    return prompt
