"""Tie sources + scoring together into a ranked recommendation."""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import FitTier, HardwareProfile, ModelCandidate, ScoredModel
from .scoring import best_runnable_variant, weights_for

CRITERIA = ["speed", "quality", "language", "context", "license"]


class Recommendation:
    def __init__(self, hw: HardwareProfile, scored: List[ScoredModel], context: int):
        self.hw = hw
        self.scored = scored
        self.context = context

    def by_tier(self, tier: FitTier) -> List[ScoredModel]:
        items = [s for s in self.scored if s.fit.tier == tier]
        return sorted(items, key=lambda s: s.overall, reverse=True)

    def top_by(self, criterion: str, n: int = 5, runnable_only: bool = True) -> List[ScoredModel]:
        pool = self.scored
        if runnable_only:
            pool = [s for s in pool if s.fit.tier != FitTier.NO_FIT]
        return sorted(pool, key=lambda s: s.scores.get(criterion, 0), reverse=True)[:n]

    def overall_top(self, n: int = 5) -> List[ScoredModel]:
        pool = [s for s in self.scored if s.fit.tier != FitTier.NO_FIT]
        return sorted(pool, key=lambda s: s.overall, reverse=True)[:n]


def recommend(
    hw: HardwareProfile,
    candidates: List[ModelCandidate],
    context: int = 8192,
    languages: Optional[List[str]] = None,
    use_case: Optional[str] = None,
    min_tps: float = 0.0,
) -> Recommendation:
    languages = languages or ["en"]
    weights = weights_for(use_case)
    scored: List[ScoredModel] = []
    for cand in candidates:
        best = best_runnable_variant(cand.variants, hw, context, languages, weights)
        if best is None:
            continue
        if min_tps > 0 and best.fit.est_tokens_per_sec < min_tps:
            continue  # --speed filter hides models below the throughput floor
        scored.append(best)
    return Recommendation(hw=hw, scored=scored, context=context)
