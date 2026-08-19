"""Multi-criteria scoring for runnable model variants.

Each criterion yields 0-100. The report shows per-criterion rankings; `overall`
is a weighted blend used only as a tie-break / default sort.
"""

from __future__ import annotations

from typing import Dict, List

from . import benchmarks
from .estimator import estimate_fit
from .models import FitResult, FitTier, HardwareProfile, ModelVariant, ScoredModel

PERMISSIVE_LICENSES = {"apache-2.0", "mit", "apache2", "bsd-3-clause", "llama3", "gemma"}

# Reference throughput (tok/s) that maps to a full speed score of 100.
SPEED_REFERENCE_TPS = 60.0
# Context length (tokens) that maps to a full context score of 100.
CONTEXT_REFERENCE = 131072

DEFAULT_WEIGHTS = {
    "speed": 0.25,
    "quality": 0.35,
    "language": 0.20,
    "context": 0.10,
    "license": 0.10,
}

USE_CASE_WEIGHTS = {
    "chat": {"speed": 0.25, "quality": 0.30, "language": 0.25, "context": 0.10, "license": 0.10},
    "coding": {"speed": 0.20, "quality": 0.45, "language": 0.10, "context": 0.15, "license": 0.10},
    "summarize": {"speed": 0.20, "quality": 0.30, "language": 0.15, "context": 0.25, "license": 0.10},
    "embedding": {"speed": 0.40, "quality": 0.30, "language": 0.15, "context": 0.05, "license": 0.10},
}


def _clamp(x: float) -> float:
    return max(0.0, min(100.0, x))


def speed_score(fit: FitResult) -> float:
    return _clamp(fit.est_tokens_per_sec / SPEED_REFERENCE_TPS * 100.0)


def quality_score(variant: ModelVariant) -> float:
    """Evidence-based quality: benchmark score discounted by confidence."""
    return _clamp(benchmarks.evaluate(variant).effective)


def language_score(variant: ModelVariant, wanted: List[str]) -> float:
    if not wanted:
        return 100.0
    supported = {l.lower() for l in variant.languages}
    if "multi" in supported:
        return 100.0
    hits = sum(1 for w in wanted if w.lower() in supported)
    return _clamp(hits / len(wanted) * 100.0)


def context_score(variant: ModelVariant) -> float:
    # Log-ish scaling so 8k isn't punished into the floor vs 128k.
    import math

    ref = math.log2(CONTEXT_REFERENCE)
    val = math.log2(max(variant.context_max, 1024))
    return _clamp(val / ref * 100.0)


def license_score(variant: ModelVariant) -> float:
    lic = variant.license.lower().strip()
    if lic in PERMISSIVE_LICENSES:
        return 100.0
    if "non-commercial" in lic or "cc-by-nc" in lic or "research" in lic:
        return 30.0
    if lic in ("unknown", ""):
        return 50.0
    return 70.0


def score_variant(
    variant: ModelVariant,
    hw: HardwareProfile,
    context: int,
    languages: List[str],
    weights: Dict[str, float],
) -> ScoredModel:
    fit = estimate_fit(variant, hw, context)
    evidence = benchmarks.evaluate(variant)
    scores = {
        "speed": round(speed_score(fit), 1),
        "quality": round(_clamp(evidence.effective), 1),
        "language": round(language_score(variant, languages), 1),
        "context": round(context_score(variant), 1),
        "license": round(license_score(variant), 1),
    }
    overall = round(sum(scores[k] * weights.get(k, 0) for k in scores), 1)
    return ScoredModel(
        variant=variant, fit=fit, scores=scores, overall=overall,
        quality_evidence=evidence.level, quality_source=evidence.source,
    )


def weights_for(use_case: str | None) -> Dict[str, float]:
    if use_case and use_case in USE_CASE_WEIGHTS:
        return USE_CASE_WEIGHTS[use_case]
    return DEFAULT_WEIGHTS


def best_runnable_variant(
    candidate_variants: List[ModelVariant],
    hw: HardwareProfile,
    context: int,
    languages: List[str],
    weights: Dict[str, float],
) -> ScoredModel | None:
    """Pick the highest-overall variant that at least TIGHT-fits.

    Falls back to the smallest variant (so NO_FIT models still appear in the
    report's NO_FIT tier) when none fit.
    """
    scored = [score_variant(v, hw, context, languages, weights) for v in candidate_variants]
    runnable = [s for s in scored if s.fit.tier != FitTier.NO_FIT]
    if runnable:
        return max(runnable, key=lambda s: s.overall)
    if scored:
        return min(scored, key=lambda s: s.fit.mem_need_gb)
    return None
