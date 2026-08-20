"""Policy audit: gate model choices against org rules for on-prem / regulated use.

`peyk audit` answers a CI-friendly question: does this machine have at least one
policy-compliant model it can actually run? Policies cover license allow-lists,
size caps, required languages, a quality floor, and whether the model must fit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .models import FitTier, ScoredModel


@dataclass
class Policy:
    max_params_b: float | None = None
    allow_licenses: set[str] | None = None   # lowercased; None = any
    require_languages: list[str] = field(default_factory=list)
    min_quality: float | None = None
    require_fit: bool = True                  # must at least TIGHT-fit

    @classmethod
    def from_file(cls, path: str) -> Policy:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        lic = raw.get("allow_licenses")
        return cls(
            max_params_b=raw.get("max_params_b"),
            allow_licenses={x.lower() for x in lic} if lic else None,
            require_languages=list(raw.get("require_languages", [])),
            min_quality=raw.get("min_quality"),
            require_fit=bool(raw.get("require_fit", True)),
        )


@dataclass
class AuditRow:
    scored: ScoredModel
    violations: list[str]

    @property
    def compliant(self) -> bool:
        return not self.violations


def check(scored: ScoredModel, policy: Policy) -> list[str]:
    v = scored.variant
    reasons: list[str] = []
    if policy.require_fit and scored.fit.tier == FitTier.NO_FIT:
        reasons.append("won't fit")
    if policy.max_params_b is not None and v.params_b > policy.max_params_b:
        reasons.append(f"params {v.params_b:g}B > {policy.max_params_b:g}B")
    if policy.allow_licenses is not None and v.license.lower() not in policy.allow_licenses:
        reasons.append(f"license '{v.license}' not allowed")
    supported = {lang.lower() for lang in v.languages}
    for lang in policy.require_languages:
        if "multi" not in supported and lang.lower() not in supported:
            reasons.append(f"missing language '{lang}'")
    if policy.min_quality is not None and scored.scores.get("quality", 0) < policy.min_quality:
        reasons.append(f"quality {scored.scores.get('quality', 0):.0f} < {policy.min_quality:g}")
    return reasons


def audit(scored_models: list[ScoredModel], policy: Policy) -> list[AuditRow]:
    rows = [AuditRow(s, check(s, policy)) for s in scored_models]
    # Compliant first, then by overall score.
    rows.sort(key=lambda r: (not r.compliant, -r.scored.overall))
    return rows


def passed(rows: list[AuditRow]) -> bool:
    """A machine passes if at least one model complies with the policy."""
    return any(r.compliant for r in rows)
