"""Reverse lookup: given a model, what hardware does it take to run it?

Inverts the fit estimator over the GPU database and the FITS/TIGHT thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .estimator import FITS_RATIO, TIGHT_RATIO, memory_need_gb
from .gpus import GPU_DB, GpuSpec
from .models import ModelVariant


@dataclass
class GpuFit:
    spec: GpuSpec
    count: int  # GPUs of this model needed to FIT


@dataclass
class PlanResult:
    variant: ModelVariant
    context: int
    mem_need_gb: float
    min_vram_fits_gb: float     # VRAM needed to comfortably fit (single device)
    min_vram_tight_gb: float
    ram_needed_gb: float        # for CPU / unified-memory hosts
    cheapest_fits: Optional[GpuSpec]
    cheapest_tight: Optional[GpuSpec]
    multi_gpu: Optional[GpuFit]  # smallest common GPU sharded to fit, if no single card does


def _smallest_fitting(need: float, ratio: float) -> Optional[GpuSpec]:
    fitting = [s for s in GPU_DB.values() if s.vram_gb * ratio >= need]
    return min(fitting, key=lambda s: s.vram_gb) if fitting else None


def _multi_gpu_option(need: float) -> Optional[GpuFit]:
    # Find the cheapest (smallest-VRAM) card that fits with 2-8 of them.
    best: Optional[GpuFit] = None
    for spec in GPU_DB.values():
        for n in range(2, 9):
            if spec.vram_gb * n * FITS_RATIO >= need:
                total = spec.vram_gb * n
                if best is None or total < best.spec.vram_gb * best.count:
                    best = GpuFit(spec, n)
                break
    return best


def plan(variant: ModelVariant, context: int = 8192) -> PlanResult:
    ctx = min(context, variant.context_max)
    need = memory_need_gb(variant, ctx)
    fits = _smallest_fitting(need, FITS_RATIO)
    tight = _smallest_fitting(need, TIGHT_RATIO)
    multi = _multi_gpu_option(need) if fits is None else None
    return PlanResult(
        variant=variant,
        context=ctx,
        mem_need_gb=round(need, 2),
        min_vram_fits_gb=round(need / FITS_RATIO, 1),
        min_vram_tight_gb=round(need / TIGHT_RATIO, 1),
        ram_needed_gb=round(need / FITS_RATIO, 1),
        cheapest_fits=fits,
        cheapest_tight=tight,
        multi_gpu=multi,
    )
