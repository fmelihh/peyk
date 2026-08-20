from peyk.models import Accelerator, HardwareProfile
from peyk.planner import plan
from peyk.resolve import resolve_variant
from peyk.snippet import build_snippets
from peyk.sources import build_catalog

CATALOG = build_catalog(offline=True)


def test_resolve_by_family_and_size():
    v = resolve_variant("llama 3.3 70b", CATALOG)
    assert v is not None
    assert v.params_b == 70
    assert "Llama" in v.family


def test_resolve_prefers_quant():
    v = resolve_variant("llama 3.1 8b", CATALOG, prefer_quant="Q8_0")
    assert v is not None and v.params_b == 8
    assert v.quant == "Q8_0"


def test_resolve_no_match():
    assert resolve_variant("nonexistent model xyz", CATALOG) is None


def test_plan_big_model_needs_big_gpu():
    v = resolve_variant("llama 3.3 70b", CATALOG)
    result = plan(v, context=8192)
    assert result.mem_need_gb > 40
    assert result.cheapest_fits is not None
    assert result.cheapest_fits.vram_gb >= 48  # a 70B Q4 won't fit a 24GB card


def test_plan_small_model_fits_small_gpu():
    v = resolve_variant("llama 3.2 3b", CATALOG)
    result = plan(v, context=8192)
    assert result.cheapest_fits is not None
    assert result.cheapest_fits.vram_gb <= 16  # 3B fits a small card


def test_snippet_ollama_for_curated():
    v = resolve_variant("qwen2.5 7b", CATALOG)
    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores_physical=8,
                         cpu_cores_logical=16, ram_total_gb=32, ram_available_gb=24,
                         accelerator=Accelerator.NONE, mem_bandwidth_gbs=50)
    data = build_snippets(v, hw)
    assert "ollama" in data["commands"]
    assert data["commands"]["ollama"].startswith("ollama run qwen2.5:7b")


def test_snippet_strips_synthetic_quant_suffix():
    from peyk.models import ModelVariant
    v = ModelVariant(model_id="llama3.1:8b-q8", family="Llama 3.1", params_b=8,
                     quant="Q8_0", file_size_gb=8.5)
    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores_physical=8,
                         cpu_cores_logical=16, ram_total_gb=32, ram_available_gb=24,
                         accelerator=Accelerator.NONE, mem_bandwidth_gbs=50)
    data = build_snippets(v, hw)
    assert data["commands"]["ollama"] == "ollama run llama3.1:8b"


def test_snippet_hf_discovered_uses_hf_commands():
    from peyk.models import ModelVariant
    v = ModelVariant(model_id="hf:someorg/CoolModel-9B-GGUF:Q4_K_M", family="CoolModel-9B",
                     params_b=9, quant="Q4_K_M", file_size_gb=5.5, source="hf-discovered")
    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores_physical=8,
                         cpu_cores_logical=16, ram_total_gb=32, ram_available_gb=24,
                         accelerator=Accelerator.NONE, mem_bandwidth_gbs=50)
    data = build_snippets(v, hw)
    assert "huggingface-cli" in data["commands"]
    assert "someorg/CoolModel-9B-GGUF" in data["commands"]["huggingface-cli"]
