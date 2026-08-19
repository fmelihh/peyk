from peyk.models import Accelerator, HardwareProfile
from peyk.profiler.probe import bandwidth_from_dimms, enrich_profile


def _cpu_profile() -> HardwareProfile:
    return HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores_physical=8, cpu_cores_logical=16,
        ram_total_gb=32.0, ram_available_gb=24.0, accelerator=Accelerator.NONE,
        mem_bandwidth_gbs=50.0, mem_bandwidth_source="estimated",
    )


def test_bandwidth_from_dimms():
    # 2x DDR4-3200 -> ~51.2 GB/s (matches real dual-channel).
    assert bandwidth_from_dimms(2, 3200) == 51.2
    assert bandwidth_from_dimms(4, 4800) == 153.6  # 4x DDR5-4800
    assert bandwidth_from_dimms(None, 3200) is None
    assert bandwidth_from_dimms(2, None) is None


def test_enrich_none_data_is_noop():
    prof = _cpu_profile()
    assert enrich_profile(prof, None) is prof


def test_enrich_measures_cpu_bandwidth_from_dimms():
    prof = _cpu_profile()
    data = {
        "cpu": {"model": "AMD Ryzen 9 7950X", "cores_physical": 16, "cores_logical": 32},
        "memory": {"type": "DDR5", "speed_mtps": 5200, "dimms_populated": 2},
        "gpus": [],
    }
    out = enrich_profile(prof, data)
    assert out.cpu_model == "AMD Ryzen 9 7950X"
    assert out.ram_type == "DDR5"
    assert out.ram_speed_mtps == 5200
    assert out.ram_channels == 2
    assert out.mem_bandwidth_gbs == round(2 * 5200 * 8 / 1000, 1)  # 83.2
    assert out.mem_bandwidth_source == "measured"


def test_enrich_apple_updates_chip_and_reestimates():
    prof = HardwareProfile(
        os="Darwin", arch="arm64", cpu_cores_physical=12, cpu_cores_logical=12,
        ram_total_gb=36.0, ram_available_gb=28.0, accelerator=Accelerator.APPLE,
        accelerator_name="Apple M3", unified_memory=True, mem_bandwidth_gbs=100.0,
    )
    data = {"cpu": {"model": "Apple M3 Max"}, "memory": {}, "gpus": []}
    out = enrich_profile(prof, data)
    assert out.accelerator_name == "Apple M3 Max"
    # Bandwidth is re-estimated from the sharper chip name (M3 Max -> ~400 GB/s).
    assert out.mem_bandwidth_gbs == 400.0
    assert out.mem_bandwidth_source == "estimated"


def test_enrich_missing_dimms_keeps_estimate():
    prof = _cpu_profile()
    data = {"cpu": {}, "memory": {"type": "DDR4"}, "gpus": []}
    out = enrich_profile(prof, data)
    assert out.ram_type == "DDR4"
    assert out.mem_bandwidth_source == "estimated"
