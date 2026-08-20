"""Model catalog sources and the assembly pipeline."""

from __future__ import annotations

from ..cache import DISCOVER_TTL, SIZES_TTL, cached_variants, key_for
from ..models import ModelCandidate, ModelVariant
from .curated import CuratedSource
from .hf_discovery import HuggingFaceDiscoverySource
from .huggingface import HuggingFaceSource
from .merge import merge_variants, to_candidates
from .ollama import OllamaSource


def build_catalog(
    offline: bool = True,
    cross_check: bool = False,
    discover: bool = False,
    use_cache: bool = True,
) -> list[ModelCandidate]:
    """Assemble the merged model catalog.

    - offline / default: curated only (deterministic, no network).
    - cross_check=True (online): enrich curated sizes via Ollama + HF.
    - discover=True (online): add trending GGUF models not in the curated set.

    Live-source results are cached under ~/.cache/peyk (TTL) unless use_cache is
    False.
    """
    curated = CuratedSource().fetch()
    results: list[list[ModelVariant]] = [curated]
    seed_ids = [v.model_id for v in curated]

    if not offline:
        if cross_check:
            results.append(cached_variants(
                key_for("ollama", seed_ids), SIZES_TTL,
                lambda: OllamaSource(seed=curated).fetch(), use_cache))
            results.append(cached_variants(
                key_for("huggingface", seed_ids), SIZES_TTL,
                lambda: HuggingFaceSource(seed=curated).fetch(), use_cache))
        if discover:
            families = [v.family for v in curated]
            results.append(cached_variants(
                key_for("hf-discovered", families), DISCOVER_TTL,
                lambda: HuggingFaceDiscoverySource(exclude_families=families).fetch(),
                use_cache))

    merged = merge_variants(results)
    return to_candidates(merged)


__all__ = [
    "CuratedSource",
    "OllamaSource",
    "HuggingFaceSource",
    "HuggingFaceDiscoverySource",
    "build_catalog",
    "merge_variants",
    "to_candidates",
]
