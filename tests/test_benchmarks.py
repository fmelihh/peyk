from peyk import benchmarks
from peyk.models import ModelVariant


def _v(family, params, quality=50.0):
    return ModelVariant(model_id="x", family=family, params_b=params,
                        file_size_gb=1.0, quality_score=quality)


def test_direct_match():
    ev = benchmarks.evaluate(_v("Qwen2.5", 7))
    assert ev.level == "direct"
    assert ev.base == 73
    assert ev.effective == 73.0  # x1.0


def test_interpolated_between_sizes():
    ev = benchmarks.evaluate(_v("Qwen2.5", 10))  # between 7 (73) and 14 (80)
    assert ev.level == "interpolated"
    assert 73 < ev.base < 80
    assert ev.effective == round(ev.base * 0.9, 1)


def test_family_extrapolation_within_2x():
    ev = benchmarks.evaluate(_v("Qwen2.5", 100))  # outside range, 100/72 < 2x
    assert ev.level == "family"
    assert ev.base == 89  # nearest (72B)


def test_repackager_guard_rejects_divergent():
    ev = benchmarks.evaluate(_v("Qwen2.5", 300, quality=44))  # 300/72 > 2x
    assert ev.level == "proxy"
    assert ev.base == 44


def test_proxy_for_unknown_family():
    ev = benchmarks.evaluate(_v("TotallyNovelModel", 9, quality=60))
    assert ev.level == "proxy"
    assert ev.source == "catalog-proxy"
    assert ev.effective == round(60 * 0.7, 1)


def test_scoring_uses_evidence():
    from peyk import scoring
    from peyk.models import Accelerator, HardwareProfile
    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores_physical=8,
                         cpu_cores_logical=16, ram_total_gb=32, ram_available_gb=24,
                         accelerator=Accelerator.NONE, mem_bandwidth_gbs=50)
    s = scoring.score_variant(_v("Qwen2.5", 7), hw, 8192, ["en"], scoring.weights_for(None))
    assert s.quality_evidence == "direct"
    assert s.scores["quality"] == 73.0
