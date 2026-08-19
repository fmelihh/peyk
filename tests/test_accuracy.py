from peyk import estimator
from peyk.cli import _parse_size_gb
from peyk.engine import recommend
from peyk.models import Accelerator, HardwareProfile, ModelVariant
from peyk.sources import build_catalog


def _hw(**kw):
    base = dict(os="Linux", arch="x86_64", cpu_cores_physical=8, cpu_cores_logical=16,
                ram_total_gb=32.0, ram_available_gb=24.0, accelerator=Accelerator.NONE,
                mem_bandwidth_gbs=50.0)
    base.update(kw)
    return HardwareProfile(**base)


def test_moe_reads_fewer_bytes_than_dense():
    moe = ModelVariant(model_id="moe", family="MoE", params_b=47, active_params_b=13,
                       file_size_gb=26.4)
    dense = ModelVariant(model_id="dense", family="Dense", params_b=47, file_size_gb=26.4)
    assert estimator._bytes_per_token(moe) < estimator._bytes_per_token(dense)


def test_moe_is_faster_than_dense_same_size():
    hw = _hw(accelerator=Accelerator.NVIDIA, vram_total_gb=48, mem_bandwidth_gbs=864)
    moe = ModelVariant(model_id="moe", family="MoE", params_b=47, active_params_b=13,
                       file_size_gb=26.4)
    dense = ModelVariant(model_id="dense", family="Dense", params_b=47, file_size_gb=26.4)
    fast = estimator.estimate_fit(moe, hw, 8192).est_tokens_per_sec
    slow = estimator.estimate_fit(dense, hw, 8192).est_tokens_per_sec
    assert fast > slow


def test_offload_blends_below_gpu_bandwidth():
    hw = _hw(accelerator=Accelerator.NVIDIA, vram_total_gb=24, mem_bandwidth_gbs=1008)
    bw = estimator._effective_bandwidth(40.0, hw)  # 40 GB need > 24 GB VRAM
    assert 50 < bw < 1008


def test_no_offload_when_it_fits_vram():
    hw = _hw(accelerator=Accelerator.NVIDIA, vram_total_gb=24, mem_bandwidth_gbs=1008)
    assert estimator._effective_bandwidth(10.0, hw) == 1008


def test_pool_cap_and_reserve():
    assert _hw(pool_cap_gb=16.0).memory_pool_gb == 16.0
    assert _hw(pool_cap_gb=16.0, reserve_gb=2.0).memory_pool_gb == 14.0
    assert _hw(reserve_gb=4.0).memory_pool_gb == round(32 * 0.8 - 4, 1)


def test_speed_filter_drops_slow_models():
    cands = build_catalog(offline=True)
    hw = _hw()  # slow CPU box
    all_models = recommend(hw, cands).scored
    fast_only = recommend(hw, cands, min_tps=20.0).scored
    assert len(fast_only) < len(all_models)
    assert all(s.fit.est_tokens_per_sec >= 20.0 for s in fast_only)


def test_parse_size_gb():
    assert _parse_size_gb("1.5GB") == 1.5
    assert _parse_size_gb("512MB") == 0.5
    assert _parse_size_gb("8") == 8.0
    assert _parse_size_gb(None) is None
