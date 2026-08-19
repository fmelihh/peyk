from peyk.engine import recommend
from peyk.models import Accelerator, HardwareProfile
from peyk.profiler import detect
from peyk.report import _disk_warning
from peyk.sources import build_catalog

CATALOG = build_catalog(offline=True)


def _hw(disk):
    return HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores_physical=8, cpu_cores_logical=16,
        ram_total_gb=64.0, ram_available_gb=48.0, disk_free_gb=disk,
        accelerator=Accelerator.NONE, mem_bandwidth_gbs=50.0,
    )


def test_detect_reports_disk():
    assert detect().disk_free_gb > 0


def test_disk_warning_when_models_exceed_free_space():
    rec = recommend(_hw(disk=3.0), CATALOG)  # only tiny models fit on disk
    warn = _disk_warning(rec)
    assert warn is not None
    assert "exceed free disk" in warn


def test_no_disk_warning_with_plenty_of_space():
    rec = recommend(_hw(disk=500.0), CATALOG)
    assert _disk_warning(rec) is None


def test_no_disk_warning_when_unknown():
    rec = recommend(_hw(disk=0.0), CATALOG)
    assert _disk_warning(rec) is None
