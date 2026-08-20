"""Evidence-based quality: map a model variant to a benchmark-backed quality
score with a confidence level, instead of a pure size heuristic.

Evidence levels (each discounts the score via a confidence multiplier):
  direct       exact family+size in the snapshot        x1.00
  interpolated same family, size between known points    x0.90
  family       same family, size extrapolated            x0.80
  proxy        no benchmark — fall back to catalog proxy  x0.70

A repackager guard prevents a model from inheriting a same-family score when its
parameter count diverges by more than 2x (protects HF-discovered repackages).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Dict, List, Optional, Tuple

from ..models import ModelVariant

_MULTIPLIER = {"live": 1.0, "direct": 1.0, "interpolated": 0.90,
               "family": 0.80, "proxy": 0.70}
_REPACKAGER_RATIO = 2.0

# Live overlay: (family_lower, params_b) -> quality. Populated by activate_live().
_LIVE: Dict[Tuple[str, float], dict] = {}


@dataclass(frozen=True)
class QualityEvidence:
    base: float          # raw quality before confidence discount (0-100)
    level: str           # direct | interpolated | family | proxy
    source: str

    @property
    def multiplier(self) -> float:
        return _MULTIPLIER[self.level]

    @property
    def effective(self) -> float:
        return round(self.base * self.multiplier, 1)


def _load() -> Tuple[Dict[str, List[dict]], str]:
    data = resources.files("peyk.benchmarks.data").joinpath("benchmarks.json")
    raw = json.loads(data.read_text(encoding="utf-8"))
    by_family: Dict[str, List[dict]] = {}
    for e in raw.get("entries", []):
        by_family.setdefault(e["family"].lower(), []).append(e)
    for entries in by_family.values():
        entries.sort(key=lambda e: e["params_b"])
    return by_family, raw.get("source", "snapshot")


_BY_FAMILY, _SOURCE = _load()


def _interpolate(entries: List[dict], p: float) -> Optional[float]:
    for lo, hi in zip(entries, entries[1:]):
        if lo["params_b"] <= p <= hi["params_b"]:
            span = hi["params_b"] - lo["params_b"]
            if span <= 0:
                return float(lo["quality"])
            t = (p - lo["params_b"]) / span
            return lo["quality"] + t * (hi["quality"] - lo["quality"])
    return None


def _divergent(a: float, b: float) -> bool:
    lo, hi = sorted((a, b))
    return lo <= 0 or hi / lo > _REPACKAGER_RATIO


def activate_live(entries: List[dict]) -> int:
    """Load live benchmark entries as an overlay taking precedence over frozen."""
    _LIVE.clear()
    for e in entries:
        _LIVE[(e["family"].lower(), float(e["params_b"]))] = e
    return len(_LIVE)


def load_live(url: Optional[str] = None, use_cache: bool = True) -> int:
    """Fetch (cached) and activate the live tier. Returns entries loaded."""
    from .. import cache
    from . import live

    target = live.resolve_url(url)
    if not target:
        return 0
    key = f"benchmarks-live:{target}"
    entries = cache.read_fresh(key, ttl=24 * 3600) if use_cache else None
    if entries is None:
        entries = live.fetch_live(url)
        if entries and use_cache:
            cache.write(key, entries)
    return activate_live(entries or [])


def evaluate(variant: ModelVariant) -> QualityEvidence:
    live_hit = _LIVE.get((variant.family.lower(), variant.params_b))
    if live_hit:
        return QualityEvidence(float(live_hit["quality"]), "live",
                               live_hit.get("source", "live"))
    entries = _BY_FAMILY.get(variant.family.lower())
    if entries:
        exact = next((e for e in entries if abs(e["params_b"] - variant.params_b) < 0.05), None)
        if exact:
            return QualityEvidence(float(exact["quality"]), "direct", _SOURCE)
        interp = _interpolate(entries, variant.params_b)
        if interp is not None:
            return QualityEvidence(round(interp, 1), "interpolated", _SOURCE)
        nearest = min(entries, key=lambda e: abs(e["params_b"] - variant.params_b))
        if not _divergent(nearest["params_b"], variant.params_b):
            return QualityEvidence(float(nearest["quality"]), "family", _SOURCE)
    # No usable benchmark evidence — fall back to the catalog proxy.
    return QualityEvidence(variant.quality_score, "proxy", "catalog-proxy")
