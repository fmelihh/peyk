from inference_profiler.models import Accelerator
from inference_profiler.profiler import accelerators, bandwidth, cpu, detect
from inference_profiler.profiler.bandwidth import estimate_bandwidth
from inference_profiler.models import HardwareProfile


def test_detect_returns_profile():
    prof = detect()
    assert prof.ram_total_gb > 0
    assert prof.cpu_cores_logical >= 1
    assert prof.mem_bandwidth_gbs > 0


def test_memory_pool_is_capacity_based_not_transient():
    # Lots of RAM in use right now, but capacity is what matters for "can run".
    prof = HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores_physical=8, cpu_cores_logical=16,
        ram_total_gb=32.0, ram_available_gb=6.0, accelerator=Accelerator.NONE,
    )
    assert prof.memory_pool_gb == 25.6  # 32 * 0.8, not 6.0


def test_memory_pool_never_below_free():
    prof = HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores_physical=8, cpu_cores_logical=16,
        ram_total_gb=16.0, ram_available_gb=15.0, accelerator=Accelerator.NONE,
    )
    assert prof.memory_pool_gb >= 15.0


def test_vram_pool_applies_reserve():
    prof = HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores_physical=8, cpu_cores_logical=16,
        ram_total_gb=64.0, ram_available_gb=48.0, accelerator=Accelerator.NVIDIA,
        vram_total_gb=24.0,
    )
    assert prof.memory_pool_gb == 21.6  # 24 * 0.9


def test_bandwidth_lookup_by_name():
    prof = HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores_physical=8, cpu_cores_logical=16,
        ram_total_gb=32, ram_available_gb=24, accelerator=Accelerator.NVIDIA,
        accelerator_name="NVIDIA GeForce RTX 4090",
    )
    assert estimate_bandwidth(prof) == 1008.0


def test_bandwidth_default_when_unknown():
    prof = HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores_physical=8, cpu_cores_logical=16,
        ram_total_gb=32, ram_available_gb=24, accelerator=Accelerator.NONE,
    )
    assert estimate_bandwidth(prof) == 50.0


def test_nvidia_parse(monkeypatch):
    monkeypatch.setattr(
        accelerators, "_run",
        lambda cmd, timeout=4: "NVIDIA GeForce RTX 4090, 24564\n",
    )
    info = accelerators._nvidia()
    assert info is not None
    assert info.kind == Accelerator.NVIDIA
    assert round(info.vram_gb) == 24


def test_nvidia_absent(monkeypatch):
    monkeypatch.setattr(accelerators, "_run", lambda cmd, timeout=4: "")
    assert accelerators._nvidia() is None


def test_cpu_flags_from_proc(monkeypatch, tmp_path):
    fake = tmp_path / "cpuinfo"
    fake.write_text("flags : fpu vme avx2 avx512f neon\n")
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(cpu, "_read_proc_cpuinfo", lambda: fake.read_text().lower())
    flags = cpu.cpu_flags()
    assert "avx2" in flags
    assert "avx512" in flags
