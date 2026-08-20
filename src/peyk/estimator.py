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
CPU_OFFLOAD_BW_GBS = 50.0  # assumed system-RAM bandwidth for offloaded layers

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
    """Active bytes read per generated token.

    Dense models read ~all weights; MoE models read only their active experts,
    so scale the on-disk size by active/total parameters.
    """
    if variant.active_params_b and variant.params_b > 0:
        frac = min(1.0, variant.active_params_b / variant.params_b)
        return variant.file_size_gb * frac
    return variant.file_size_gb


def _effective_bandwidth(mem_need: float, hw: HardwareProfile) -> float:
    """Blend GPU + system-RAM bandwidth when a model spills out of VRAM.

    For a discrete accelerator whose weights don't all fit in VRAM, the
    offloaded fraction is served at slower system-RAM bandwidth; time adds, so
    the effective bandwidth is the harmonic blend.
    """
    gpu_bw = hw.mem_bandwidth_gbs
    if hw.unified_memory or hw.accelerator == Accelerator.NONE or hw.vram_total_gb <= 0:
        return gpu_bw
    if mem_need <= hw.vram_total_gb:
        return gpu_bw
    frac_gpu = max(0.0, min(1.0, hw.vram_total_gb / mem_need))
    frac_cpu = 1.0 - frac_gpu
    denom = frac_gpu / max(gpu_bw, 1e-6) + frac_cpu / CPU_OFFLOAD_BW_GBS
    return 1.0 / denom if denom > 0 else gpu_bw


def estimate_tokens_per_sec(
    variant: ModelVariant, hw: HardwareProfile, effective_bw: float | None = None
) -> float:
    """Bandwidth-bound throughput estimate. Coarse by design."""
    bw = hw.mem_bandwidth_gbs if effective_bw is None else effective_bw
    if bw <= 0:
        return 0.0
    mult = BACKEND_MULTIPLIER.get(hw.accelerator, 0.35)
    per_token_gb = max(_bytes_per_token(variant), 1e-6)
    return bw / per_token_gb * mult


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
    kv = kv_cache_gb(ctx, variant)
    mem_need = variant.file_size_gb + kv + RUNTIME_OVERHEAD_GB
    tier = classify(mem_need, hw.memory_pool_gb)
    tps = estimate_tokens_per_sec(variant, hw, _effective_bandwidth(mem_need, hw))
    return FitResult(
        variant=variant,
        mem_need_gb=round(mem_need, 2),
        tier=tier,
        est_tokens_per_sec=round(tps, 1),
        weights_gb=round(variant.file_size_gb, 2),
        kv_cache_gb=round(kv, 2),
        overhead_gb=RUNTIME_OVERHEAD_GB,
    )
