"""Model catalog sources and the assembly pipeline."""

from __future__ import annotations

from typing import List

from ..models import ModelCandidate, ModelVariant
from .curated import CuratedSource
from .huggingface import HuggingFaceSource
from .merge import merge_variants, to_candidates
from .ollama import OllamaSource


def build_catalog(offline: bool = True, cross_check: bool = False) -> List[ModelCandidate]:
    """Assemble the merged model catalog.

    - offline / default: curated only (deterministic, no network).
    - cross_check=True and not offline: enrich curated sizes via Ollama + HF.
    """
    curated = CuratedSource().fetch()
    results: List[List[ModelVariant]] = [curated]

    if not offline and cross_check:
        results.append(OllamaSource(seed=curated).fetch())
        results.append(HuggingFaceSource(seed=curated).fetch())

    merged = merge_variants(results)
    return to_candidates(merged)


__all__ = [
    "CuratedSource",
    "OllamaSource",
    "HuggingFaceSource",
    "build_catalog",
    "merge_variants",
    "to_candidates",
]
