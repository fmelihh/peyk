"""Estimate effective memory bandwidth (GB/s) — the main driver of token speed.

Discrete-GPU bandwidth comes from the shared GPU database (`peyk.gpus`); Apple
unified-memory chips use the table below. Exact bandwidth is hard to detect
portably, so this is an estimate, surfaced as such in the report.
"""

from __future__ import annotations

from ..gpus import GPU_DB
from ..models import Accelerator, HardwareProfile

# Apple unified memory bandwidth by chip name fragment.
_APPLE_TABLE = {
    "m1 max": 400, "m1 pro": 200, "m1 ultra": 800, "m1": 68,
    "m2 max": 400, "m2 pro": 200, "m2 ultra": 800, "m2": 100,
    "m3 max": 400, "m3 pro": 150, "m3 ultra": 800, "m3": 100,
    "m4 max": 546, "m4 pro": 273, "m4": 120,
}

_DEFAULTS = {
    Accelerator.NVIDIA: 600.0,
    Accelerator.APPLE: 100.0,
    Accelerator.AMD: 500.0,
    Accelerator.NONE: 50.0,  # typical dual-channel DDR4/5 desktop RAM
}


def estimate_bandwidth(profile: HardwareProfile) -> float:
    name = (profile.accelerator_name or "").lower()

    if profile.accelerator == Accelerator.APPLE:
        for frag, bw in _APPLE_TABLE.items():
            if frag in name:
                return float(bw)

    # Discrete GPU: match the longest GPU-DB key contained in the name.
    matches = [k for k in GPU_DB if k in name]
    if matches:
        return GPU_DB[max(matches, key=len)].bandwidth_gbs

    return _DEFAULTS.get(profile.accelerator, 50.0)
