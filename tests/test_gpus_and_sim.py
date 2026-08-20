import pytest

from peyk.gpus import lookup_gpu, parse_gpu_arg
from peyk.models import Accelerator
from peyk.simulate import simulate_profile


def _base():
    from peyk.models import HardwareProfile
    return HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores_physical=8, cpu_cores_logical=16,
        ram_total_gb=32.0, ram_available_gb=24.0, accelerator=Accelerator.NONE,
        mem_bandwidth_gbs=50.0,
    )


def test_lookup_exact_and_fuzzy():
    assert lookup_gpu("rtx 4090").vram_gb == 24
    assert lookup_gpu("NVIDIA GeForce RTX 4090").name == "rtx 4090"  # substring
    assert lookup_gpu("totally unknown card") is None


def test_parse_count_prefix():
    spec, count = parse_gpu_arg("2x RTX 5090")
    assert count == 2
    assert spec.name == "rtx 5090"
    assert spec.vram_gb == 32


def test_parse_inline_vram_override():
    spec, count = parse_gpu_arg("A100 40GB")
    assert count == 1
    assert spec.vram_gb == 40  # DB has an explicit a100 40gb entry


def test_parse_unknown_raises_with_suggestion():
    with pytest.raises(ValueError) as exc:
        parse_gpu_arg("RTX 9090")
    assert "Unknown GPU" in str(exc.value)


def test_simulate_single_gpu():
    prof = simulate_profile("RTX 4090", base=_base())
    assert prof.simulated is True
    assert prof.accelerator == Accelerator.NVIDIA
    assert prof.vram_total_gb == 24
    assert prof.gpu_count == 1
    assert prof.mem_bandwidth_source == "simulated"
    assert prof.mem_bandwidth_gbs == 1008
    # keeps host CPU/RAM
    assert prof.ram_total_gb == 32.0


def test_simulate_multi_gpu_sums_vram():
    prof = simulate_profile("2x RTX 5090", base=_base())
    assert prof.vram_total_gb == 64
    assert prof.gpu_count == 2
    assert "2x" in prof.accelerator_name


def test_gpu_only_uses_full_vram():
    normal = simulate_profile("RTX 4090", base=_base())
    full = simulate_profile("RTX 4090", gpu_only=True, base=_base())
    assert full.memory_pool_gb > normal.memory_pool_gb
    assert full.memory_pool_gb == 24.0  # no reserve
