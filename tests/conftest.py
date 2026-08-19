import pytest

from inference_profiler.models import Accelerator, HardwareProfile, ModelVariant


@pytest.fixture
def laptop_cpu() -> HardwareProfile:
    return HardwareProfile(
        os="Linux", arch="x86_64",
        cpu_cores_physical=4, cpu_cores_logical=8, cpu_flags=["avx2"],
        ram_total_gb=16.0, ram_available_gb=12.0,
        accelerator=Accelerator.NONE, mem_bandwidth_gbs=50.0,
    )


@pytest.fixture
def rtx4090() -> HardwareProfile:
    return HardwareProfile(
        os="Linux", arch="x86_64",
        cpu_cores_physical=16, cpu_cores_logical=32, cpu_flags=["avx512", "avx2"],
        ram_total_gb=64.0, ram_available_gb=48.0,
        accelerator=Accelerator.NVIDIA, accelerator_name="NVIDIA GeForce RTX 4090",
        vram_total_gb=24.0, mem_bandwidth_gbs=1008.0,
    )


@pytest.fixture
def mac_m3() -> HardwareProfile:
    return HardwareProfile(
        os="Darwin", arch="arm64",
        cpu_cores_physical=12, cpu_cores_logical=12, cpu_flags=["neon"],
        ram_total_gb=36.0, ram_available_gb=28.0,
        accelerator=Accelerator.APPLE, accelerator_name="Apple M3 Max",
        vram_total_gb=0.0, unified_memory=True, mem_bandwidth_gbs=400.0,
    )


@pytest.fixture
def small_variant() -> ModelVariant:
    return ModelVariant(
        model_id="llama3.2:3b", family="Llama 3.2", params_b=3,
        quant="Q4_K_M", file_size_gb=2.0, context_max=131072,
        languages=["en"], license="llama3", quality_score=58,
    )


@pytest.fixture
def big_variant() -> ModelVariant:
    return ModelVariant(
        model_id="llama3.3:70b", family="Llama 3.3", params_b=70,
        quant="Q4_K_M", file_size_gb=43.0, context_max=131072,
        languages=["en"], license="llama3", quality_score=90,
    )
