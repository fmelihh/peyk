"""Accelerator detection: NVIDIA, Apple Silicon, AMD. All best-effort."""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass

from ..models import Accelerator

GB = 1024 ** 3


@dataclass
class AccelInfo:
    kind: Accelerator = Accelerator.NONE
    name: str | None = None
    vram_gb: float = 0.0  # aggregate across all GPUs
    unified: bool = False
    count: int = 0
    driver: str | None = None
    compute_cap: str | None = None


def _run(cmd: list[str], timeout: int = 4) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _nvidia() -> AccelInfo | None:
    out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
         "--format=csv,noheader,nounits"]
    )
    if not out.strip():
        return None
    # Aggregate VRAM across all GPUs — multi-GPU servers can shard a model.
    total_vram, count, first_name, driver, cap = 0.0, 0, None, None, None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            vram_mib = float(parts[1])
        except ValueError:
            continue
        total_vram += vram_mib / 1024
        count += 1
        if first_name is None:
            first_name = parts[0]
        if driver is None and len(parts) >= 3:
            driver = parts[2] or None
        if cap is None and len(parts) >= 4:
            cap = parts[3] or None
    if count == 0:
        return None
    name = first_name if count == 1 else f"{count}x {first_name}"
    return AccelInfo(
        Accelerator.NVIDIA, name, round(total_vram, 1), unified=False, count=count,
        driver=driver, compute_cap=cap,
    )


def _apple() -> AccelInfo | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    name = "Apple Silicon"
    out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if out.strip():
        name = out.strip()
    # Unified memory: VRAM pool == system RAM; report 0 here and let the
    # profile treat it as unified so memory_pool falls back to RAM.
    return AccelInfo(Accelerator.APPLE, name, vram_gb=0.0, unified=True, count=1)


def _amd() -> AccelInfo | None:
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if not out.strip():
        return None
    vram_gb = 0.0
    for m in re.finditer(r"(\d+)\s*(?:bytes)?", out):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if val > 1e8:  # looks like a byte count
            vram_gb = max(vram_gb, round(val / GB, 1))
    name = "AMD GPU"
    return AccelInfo(Accelerator.AMD, name, vram_gb, unified=False, count=1)


def detect_accelerator() -> AccelInfo:
    for probe in (_nvidia, _apple, _amd):
        info = probe()
        if info is not None:
            return info
    return AccelInfo()
