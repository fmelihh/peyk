"""Curated GPU spec database, shared by bandwidth lookup, `--gpu` simulation,
and reverse lookup (`peyk plan`).

Each entry maps a canonical name to its VRAM and effective memory bandwidth —
the two numbers that decide "does it fit" and "how fast". Discrete GPUs only;
Apple unified-memory chips are handled separately (their pool is system RAM).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .models import Accelerator


@dataclass(frozen=True)
class GpuSpec:
    name: str
    vendor: Accelerator
    vram_gb: float
    bandwidth_gbs: float


def _nv(name: str, vram: float, bw: float) -> GpuSpec:
    return GpuSpec(name, Accelerator.NVIDIA, vram, bw)


def _amd(name: str, vram: float, bw: float) -> GpuSpec:
    return GpuSpec(name, Accelerator.AMD, vram, bw)


# Canonical, lowercased keys. Order matters only for close-match suggestions.
GPU_DB: dict[str, GpuSpec] = {spec.name: spec for spec in [
    # NVIDIA consumer (Ada / Ampere / Blackwell)
    _nv("rtx 5090", 32, 1790),
    _nv("rtx 5080", 16, 960),
    _nv("rtx 4090", 24, 1008),
    _nv("rtx 4080", 16, 717),
    _nv("rtx 4070 ti", 12, 504),
    _nv("rtx 4070", 12, 504),
    _nv("rtx 4060 ti", 16, 288),
    _nv("rtx 4060", 8, 272),
    _nv("rtx 3090 ti", 24, 1008),
    _nv("rtx 3090", 24, 936),
    _nv("rtx 3080", 10, 760),
    _nv("rtx 3070", 8, 448),
    _nv("rtx 3060", 12, 360),
    # NVIDIA datacenter / pro
    _nv("h200", 141, 4800),
    _nv("h100", 80, 3350),
    _nv("a100 80gb", 80, 2039),
    _nv("a100 40gb", 40, 1555),
    _nv("a100", 80, 2039),
    _nv("l40s", 48, 864),
    _nv("l40", 48, 864),
    _nv("a6000", 48, 768),
    _nv("a40", 48, 696),
    _nv("a5000", 24, 768),
    _nv("v100", 32, 900),
    _nv("t4", 16, 320),
    # AMD
    _amd("mi300x", 192, 5300),
    _amd("rx 7900 xtx", 24, 960),
    _amd("rx 7900 xt", 20, 800),
]}

_COUNT_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(.+)$")
_VRAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb", re.IGNORECASE)


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def lookup_gpu(name: str) -> GpuSpec | None:
    """Find a GPU by (fuzzy) name. Longest matching canonical key wins."""
    n = _normalize(name)
    if n in GPU_DB:
        return GPU_DB[n]
    # Substring either direction; prefer the most specific (longest) key.
    candidates = [k for k in GPU_DB if k in n or n in k]
    if candidates:
        best = max(candidates, key=len)
        return GPU_DB[best]
    return None


def suggest(name: str, n: int = 3) -> list[str]:
    return difflib.get_close_matches(_normalize(name), list(GPU_DB), n=n, cutoff=0.3)


def parse_gpu_arg(arg: str) -> tuple[GpuSpec, int]:
    """Parse a `--gpu` value like "RTX 4090", "2x RTX 5090", or "A100 80GB".

    Returns (spec, count). Raises ValueError (with suggestions) on no match.
    """
    count = 1
    rest = arg
    m = _COUNT_RE.match(arg)
    if m:
        count = max(1, int(m.group(1)))
        rest = m.group(2)

    spec = lookup_gpu(rest)
    vram_override = None
    if spec is None:
        vm = _VRAM_RE.search(rest)
        if vm:
            vram_override = float(vm.group(1))
            spec = lookup_gpu(_VRAM_RE.sub("", rest))
    if spec is None:
        hint = suggest(rest)
        extra = f" Did you mean: {', '.join(hint)}?" if hint else ""
        raise ValueError(f"Unknown GPU '{arg}'.{extra}")
    if vram_override is not None:
        spec = GpuSpec(spec.name, spec.vendor, vram_override, spec.bandwidth_gbs)
    return spec, count
