"""Query transformation modules."""

from src.query.base import QueryExpanderLike
from src.query.decomposition import (
    DEFAULT_DECOMPOSITION_SYSTEM_PROMPT,
    DEFAULT_DECOMPOSITION_USER_PROMPT_TEMPLATE,
    HotpotDecomposeExpander,
)
from src.query.factory import build_query_expander, resolve_query_expansion_mode
from src.query.hyde import (
    DEFAULT_HYDE_SYSTEM_PROMPT,
    DEFAULT_HYDE_USER_PROMPT_TEMPLATE,
    HOTPOT_HYDE_SYSTEM_PROMPT,
    HOTPOT_HYDE_USER_PROMPT_TEMPLATE,
    HyDEExpander,
)

__all__ = [
    "DEFAULT_DECOMPOSITION_SYSTEM_PROMPT",
    "DEFAULT_DECOMPOSITION_USER_PROMPT_TEMPLATE",
    "DEFAULT_HYDE_SYSTEM_PROMPT",
    "DEFAULT_HYDE_USER_PROMPT_TEMPLATE",
    "HotpotDecomposeExpander",
    "HyDEExpander",
    "HOTPOT_HYDE_SYSTEM_PROMPT",
    "HOTPOT_HYDE_USER_PROMPT_TEMPLATE",
    "QueryExpanderLike",
    "build_query_expander",
    "resolve_query_expansion_mode",
]
