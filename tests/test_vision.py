from peyk import scoring
from peyk.engine import recommend
from peyk.models import Accelerator, HardwareProfile
from peyk.sources import build_catalog

CATALOG = build_catalog(offline=True)


def _hw():
    return HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores_physical=16, cpu_cores_logical=32,
        ram_total_gb=64.0, ram_available_gb=48.0, accelerator=Accelerator.NVIDIA,
        accelerator_name="RTX 4090", vram_total_gb=24.0, mem_bandwidth_gbs=1008.0,
    )


def test_vision_usecase_returns_only_vision_models():
    rec = recommend(_hw(), CATALOG, use_case="vision")
    assert rec.scored
    families = {s.variant.family for s in rec.scored}
    assert "Qwen2.5-VL" in families or "Gemma 3" in families
    # No text-only families should appear.
    assert "Mistral" not in families
    assert all(s.variant.modality == "vision" for s in rec.scored)


def test_non_vision_usecase_includes_text_models():
    rec = recommend(_hw(), CATALOG, use_case="chat")
    families = {s.variant.family for s in rec.scored}
    assert "Qwen2.5" in families  # text model present


def test_weights_for_vision_and_math():
    assert scoring.weights_for("vision")["quality"] == 0.40
    assert scoring.weights_for("math")["quality"] == 0.50


def test_vision_models_have_benchmark_evidence():
    from peyk import benchmarks
    from peyk.models import ModelVariant
    v = ModelVariant(model_id="qwen2.5vl:7b", family="Qwen2.5-VL", params_b=7,
                     file_size_gb=5.0, modality="vision")
    assert benchmarks.evaluate(v).level == "direct"
