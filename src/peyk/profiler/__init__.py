"""Hardware detection. Linux-first, degrades gracefully on macOS / Windows.

The public entry point is `detect()`. Everything else is best-effort and
returns partial data rather than raising, so a report is always producible.
"""

from __future__ import annotations

import os
import platform

import psutil

from ..models import Accelerator, HardwareProfile
from .accelerators import detect_accelerator
from .bandwidth import estimate_bandwidth
from .cpu import cpu_flags
from .probe import enrich_profile, run_probe

GB = 1024 ** 3


def _numa_nodes() -> int | None:
    """Count NUMA nodes on Linux multi-socket boxes; None elsewhere."""
    try:
        base = "/sys/devices/system/node"
        nodes = [d for d in os.listdir(base) if d.startswith("node") and d[4:].isdigit()]
        return len(nodes) or None
    except OSError:
        return None


def detect(deep: bool = False, allow_sudo: bool = False) -> HardwareProfile:
    vm = psutil.virtual_memory()
    ram_total = round(vm.total / GB, 1)
    ram_avail = round(vm.available / GB, 1)
    try:
        swap_total = round(psutil.swap_memory().total / GB, 1)
    except OSError:
        swap_total = 0.0

    # Free space where models are typically downloaded (home dir).
    try:
        disk_free = round(psutil.disk_usage(os.path.expanduser("~")).free / GB, 1)
    except OSError:
        disk_free = 0.0

    acc = detect_accelerator()

    profile = HardwareProfile(
        os=platform.system(),
        arch=platform.machine(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        cpu_flags=cpu_flags(),
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        swap_total_gb=swap_total,
        disk_free_gb=disk_free,
        numa_nodes=_numa_nodes(),
        accelerator=acc.kind,
        accelerator_name=acc.name,
        gpu_count=acc.count,
        gpu_driver=acc.driver,
        vram_total_gb=acc.vram_gb,
        unified_memory=acc.unified,
    )
    profile.mem_bandwidth_gbs = estimate_bandwidth(profile)

    if deep:
        profile = enrich_profile(profile, run_probe(allow_sudo=allow_sudo))
    return profile


__all__ = ["detect", "HardwareProfile", "Accelerator"]
