"""Resolve a free-text model query to a catalog variant.

Used by `peyk plan` and `peyk snippet`. Matches on family-token overlap and,
when the query names a size (e.g. "70b"), the parameter count.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .models import ModelCandidate, ModelVariant

_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
_TOKEN_RE = re.compile(r"[a-z0-9.]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _query_params(query: str) -> Optional[float]:
    m = _PARAMS_RE.search(query)
    return float(m.group(1)) if m else None


def _score(query: str, variant: ModelVariant) -> float:
    q_tokens = _tokens(query)
    hay = _tokens(f"{variant.family} {variant.model_id}")
    overlap = len(q_tokens & hay)
    score = float(overlap)
    qp = _query_params(query)
    if qp is not None:
        # Exact size match is a strong signal; near miss is penalized.
        score += 3.0 if abs(variant.params_b - qp) < 0.05 else -abs(variant.params_b - qp) * 0.1
    return score


def resolve_variant(
    query: str, candidates: List[ModelCandidate], prefer_quant: str = "Q4_K_M"
) -> Optional[ModelVariant]:
    """Return the best-matching variant, preferring a common quant on ties."""
    variants = [v for c in candidates for v in c.variants]
    if not variants:
        return None
    best_score = max(_score(query, v) for v in variants)
    if best_score <= 0:
        return None
    top = [v for v in variants if _score(query, v) == best_score]
    # Prefer the requested quant, else the largest file (usually higher quality).
    preferred = [v for v in top if v.quant.lower() == prefer_quant.lower()]
    pool = preferred or top
    return max(pool, key=lambda v: v.file_size_gb)
