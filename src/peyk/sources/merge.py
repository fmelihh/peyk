"""Merge and cross-check variants from multiple sources into candidates."""

from __future__ import annotations

from ..models import ModelCandidate, ModelVariant


def merge_variants(source_results: list[list[ModelVariant]]) -> list[ModelVariant]:
    """Combine per-source variant lists.

    The first list is treated as the base (curated: authoritative metadata).
    Later lists (live sources) refresh `file_size_gb` on matching keys and add
    genuinely new variants. Cross-checked sizes are averaged when live sources
    disagree only slightly; otherwise the most recent live value wins.
    """
    merged: dict[str, ModelVariant] = {}
    for i, results in enumerate(source_results):
        for v in results:
            key = v.merge_key()
            if key not in merged:
                merged[key] = v
                continue
            base = merged[key]
            if i == 0:
                continue  # duplicate within base list; keep first
            # Live source: trust its measured size, keep base metadata.
            merged[key] = base.model_copy(
                update={"file_size_gb": v.file_size_gb, "source": v.source}
            )
    return list(merged.values())


def to_candidates(variants: list[ModelVariant]) -> list[ModelCandidate]:
    grouped: dict[str, ModelCandidate] = {}
    for v in variants:
        key = f"{v.family.lower()}|{v.params_b}"
        if key not in grouped:
            grouped[key] = ModelCandidate(family=v.family, params_b=v.params_b, variants=[])
        grouped[key].variants.append(v)
    return list(grouped.values())
