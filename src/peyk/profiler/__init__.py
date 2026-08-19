"""Hardware detection. Linux-first, degrades gracefully on macOS / Windows.

The public entry point is `detect()`. Everything else is best-effort and
returns partial data rather than raising, so a report is always producible.
"""

from __future__ import annotations

import platform
from typing import List

import psutil

from ..models import Accelerator, HardwareProfile
from .accelerators import detect_accelerator
from .bandwidth import estimate_bandwidth
from .cpu import cpu_flags

GB = 1024 ** 3


def detect() -> HardwareProfile:
    vm = psutil.virtual_memory()
    ram_total = round(vm.total / GB, 1)
    ram_avail = round(vm.available / GB, 1)

    acc = detect_accelerator()

    profile = HardwareProfile(
        os=platform.system(),
        arch=platform.machine(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        cpu_flags=cpu_flags(),
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        accelerator=acc.kind,
        accelerator_name=acc.name,
        gpu_count=acc.count,
        vram_total_gb=acc.vram_gb,
        unified_memory=acc.unified,
    )
    profile.mem_bandwidth_gbs = estimate_bandwidth(profile)
    return profile


__all__ = ["detect", "HardwareProfile", "Accelerator"]
