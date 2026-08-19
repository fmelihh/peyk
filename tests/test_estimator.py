from inference_profiler import estimator
from inference_profiler.models import FitTier, ModelVariant


def test_kv_cache_grows_with_context(small_variant):
    small = estimator.kv_cache_gb(2048, small_variant)
    large = estimator.kv_cache_gb(32768, small_variant)
    assert large > small > 0


def test_kv_cache_uses_arch_hints_when_present():
    v = ModelVariant(model_id="x:7b", family="X", params_b=7, file_size_gb=4.0,
                     n_layers=32, hidden=4096, gqa_factor=1.0)
    # 2*32*4096*2*4096 bytes / 1e9 = ~2.15 GB at full MHA
    assert 1.8 < estimator.kv_cache_gb(4096, v) < 2.5


def test_memory_need_includes_overhead(small_variant):
    need = estimator.memory_need_gb(small_variant, 8192)
    assert need > small_variant.file_size_gb + estimator.RUNTIME_OVERHEAD_GB - 0.01


def test_classify_thresholds():
    assert estimator.classify(6.0, 10.0) == FitTier.FITS      # 60%
    assert estimator.classify(9.0, 10.0) == FitTier.TIGHT     # 90%
    assert estimator.classify(9.9, 10.0) == FitTier.NO_FIT    # 99%
    assert estimator.classify(5.0, 0.0) == FitTier.NO_FIT


def test_small_model_fits_cpu_laptop(laptop_cpu, small_variant):
    fit = estimator.estimate_fit(small_variant, laptop_cpu, 8192)
    assert fit.tier == FitTier.FITS


def test_big_model_does_not_fit_laptop(laptop_cpu, big_variant):
    fit = estimator.estimate_fit(big_variant, laptop_cpu, 8192)
    assert fit.tier == FitTier.NO_FIT


def test_big_model_fits_when_pool_is_large(mac_m3, big_variant):
    # 28 GB pool: 70B Q4 (~43 GB) still won't fit.
    assert estimator.estimate_fit(big_variant, mac_m3, 8192).tier == FitTier.NO_FIT


def test_speed_scales_with_bandwidth(small_variant, laptop_cpu, rtx4090):
    slow = estimator.estimate_fit(small_variant, laptop_cpu, 8192).est_tokens_per_sec
    fast = estimator.estimate_fit(small_variant, rtx4090, 8192).est_tokens_per_sec
    assert fast > slow > 0


def test_context_capped_at_variant_max(small_variant, laptop_cpu):
    v = small_variant.model_copy(update={"context_max": 4096})
    # Requesting 200k should be capped to 4096 for the KV estimate.
    fit = estimator.estimate_fit(v, laptop_cpu, 200000)
    capped = estimator.memory_need_gb(v, 4096)
    assert abs(fit.mem_need_gb - round(capped, 2)) < 0.01
