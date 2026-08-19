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
    vram_gb: float = 0.0
    unified: bool = False


def _run(cmd: list[str], timeout: int = 4) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _nvidia() -> AccelInfo | None:
    out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if not out.strip():
        return None
    # Take the first (or largest) GPU.
    best_name, best_vram = None, 0.0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            vram_mib = float(parts[1])
        except ValueError:
            continue
        vram_gb = round(vram_mib / 1024, 1)
        if vram_gb > best_vram:
            best_name, best_vram = name, vram_gb
    if best_name is None:
        return None
    return AccelInfo(Accelerator.NVIDIA, best_name, best_vram, unified=False)


def _apple() -> AccelInfo | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    name = "Apple Silicon"
    out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if out.strip():
        name = out.strip()
    # Unified memory: VRAM pool == system RAM; report 0 here and let the
    # profile treat it as unified so memory_pool falls back to RAM.
    return AccelInfo(Accelerator.APPLE, name, vram_gb=0.0, unified=True)


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
    return AccelInfo(Accelerator.AMD, name, vram_gb, unified=False)


def detect_accelerator() -> AccelInfo:
    for probe in (_nvidia, _apple, _amd):
        info = probe()
        if info is not None:
            return info
    return AccelInfo()
