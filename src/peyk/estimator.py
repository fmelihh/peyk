"""Memory-fit and rough speed estimation for a model variant on a machine.

All formulas here are deliberately coarse heuristics. Speed figures are labelled
estimates in the report, never guarantees.
"""

from __future__ import annotations

from .models import Accelerator, FitResult, FitTier, HardwareProfile, ModelVariant

# --- tunable constants (documented in docs/design.md) ---
FITS_RATIO = 0.70  # mem_need <= 70% of pool  -> comfortable
TIGHT_RATIO = 0.95  # mem_need <= 95% of pool  -> works but strained
RUNTIME_OVERHEAD_GB = 0.9  # backend/context buffers outside weights + KV
KV_PER_TOKEN_PARAM = 7e-5  # GB per (token * params_b) at fp16, MHA baseline

# Rough throughput multipliers relative to raw memory-bandwidth bound.
BACKEND_MULTIPLIER = {
    Accelerator.NVIDIA: 0.85,
    Accelerator.APPLE: 0.55,
    Accelerator.AMD: 0.50,
    Accelerator.NONE: 0.35,
}


def kv_cache_gb(context: int, variant: ModelVariant) -> float:
    """Estimate fp16 KV-cache size for a full context window.

    Prefers real architecture numbers when the catalog provides them, otherwise
    falls back to a params-scaled, GQA-aware approximation.
    """
    if variant.n_layers and variant.hidden:
        # 2 (K+V) * layers * hidden * 2 bytes * context, scaled by GQA sharing.
        bytes_total = 2 * variant.n_layers * variant.hidden * 2 * context
        return bytes_total / 1e9 * variant.gqa_factor
    return context * variant.params_b * KV_PER_TOKEN_PARAM * variant.gqa_factor


def memory_need_gb(variant: ModelVariant, context: int) -> float:
    """Total memory to load weights + KV cache + runtime overhead."""
    return variant.file_size_gb + kv_cache_gb(context, variant) + RUNTIME_OVERHEAD_GB


def _bytes_per_token(variant: ModelVariant) -> float:
    """Active bytes read per generated token ~= quantized weight size."""
    return variant.file_size_gb  # dense models read ~all weights per token


def estimate_tokens_per_sec(variant: ModelVariant, hw: HardwareProfile) -> float:
    """Bandwidth-bound throughput estimate. Coarse by design."""
    if hw.mem_bandwidth_gbs <= 0:
        return 0.0
    mult = BACKEND_MULTIPLIER.get(hw.accelerator, 0.35)
    per_token_gb = max(_bytes_per_token(variant), 1e-6)
    return hw.mem_bandwidth_gbs / per_token_gb * mult


def classify(mem_need: float, pool_gb: float) -> FitTier:
    if pool_gb <= 0:
        return FitTier.NO_FIT
    if mem_need <= FITS_RATIO * pool_gb:
        return FitTier.FITS
    if mem_need <= TIGHT_RATIO * pool_gb:
        return FitTier.TIGHT
    return FitTier.NO_FIT


def estimate_fit(variant: ModelVariant, hw: HardwareProfile, context: int) -> FitResult:
    ctx = min(context, variant.context_max)
    mem_need = memory_need_gb(variant, ctx)
    tier = classify(mem_need, hw.memory_pool_gb)
    tps = estimate_tokens_per_sec(variant, hw)
    return FitResult(
        variant=variant,
        mem_need_gb=round(mem_need, 2),
        tier=tier,
        est_tokens_per_sec=round(tps, 1),
    )
