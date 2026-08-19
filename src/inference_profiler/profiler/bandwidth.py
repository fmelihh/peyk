"""Estimate effective memory bandwidth (GB/s) — the main driver of token speed.

Exact bandwidth is hard to detect portably, so we use a lookup keyed on the
detected accelerator/name with sane fallbacks. This is an estimate, surfaced as
such in the report.
"""

from __future__ import annotations

from ..models import Accelerator, HardwareProfile

# Rough on-device memory bandwidth by known name fragments.
_NAME_TABLE = {
    # NVIDIA
    "h100": 3350, "a100": 2000, "4090": 1008, "4080": 717, "3090": 936,
    "3080": 760, "4070": 504, "a6000": 768, "l40": 864, "5090": 1790,
    # Apple unified memory
    "m1 max": 400, "m1 pro": 200, "m1 ultra": 800, "m1": 68,
    "m2 max": 400, "m2 pro": 200, "m2 ultra": 800, "m2": 100,
    "m3 max": 400, "m3 pro": 150, "m3 ultra": 800, "m3": 100,
    "m4 max": 546, "m4 pro": 273, "m4": 120,
    # AMD
    "mi300": 5300, "7900": 960,
}

_DEFAULTS = {
    Accelerator.NVIDIA: 600.0,
    Accelerator.APPLE: 100.0,
    Accelerator.AMD: 500.0,
    Accelerator.NONE: 50.0,  # typical dual-channel DDR4/5 desktop RAM
}


def estimate_bandwidth(profile: HardwareProfile) -> float:
    name = (profile.accelerator_name or "").lower()
    for frag, bw in _NAME_TABLE.items():
        if frag in name:
            return float(bw)
    return _DEFAULTS.get(profile.accelerator, 50.0)
