import httpx
import pytest

from peyk.models import ModelVariant
from peyk.sources import build_catalog
from peyk.sources.curated import CuratedSource
from peyk.sources.huggingface import HuggingFaceSource
from peyk.sources.merge import merge_variants, to_candidates
from peyk.sources.ollama import OllamaSource


def test_curated_loads_variants():
    variants = CuratedSource().fetch()
    assert len(variants) > 10
    assert all(isinstance(v, ModelVariant) for v in variants)
    families = {v.family for v in variants}
    assert "Qwen2.5" in families


def test_build_catalog_offline_groups_by_family_size():
    cands = build_catalog(offline=True)
    assert cands
    # Llama 3.1 8B has two quant variants in the catalog -> one candidate, 2 variants.
    llama = [c for c in cands if c.family == "Llama 3.1" and c.params_b == 8]
    assert len(llama) == 1
    assert len(llama[0].variants) >= 2


def test_merge_prefers_live_size_over_curated():
    curated = [ModelVariant(model_id="m:7b", family="M", params_b=7,
                            quant="Q4_K_M", file_size_gb=4.5, quality_score=70)]
    live = [curated[0].model_copy(update={"file_size_gb": 4.12, "source": "ollama"})]
    merged = merge_variants([curated, live])
    assert len(merged) == 1
    assert merged[0].file_size_gb == 4.12
    assert merged[0].source == "ollama"
    assert merged[0].quality_score == 70  # metadata preserved from curated


def test_to_candidates_dedup_by_family_and_size():
    variants = [
        ModelVariant(model_id="a:7b", family="A", params_b=7, quant="Q4_K_M", file_size_gb=4),
        ModelVariant(model_id="a:7b-q8", family="A", params_b=7, quant="Q8_0", file_size_gb=7),
    ]
    cands = to_candidates(variants)
    assert len(cands) == 1
    assert len(cands[0].variants) == 2


def _mock_transport_ollama(size_bytes: int):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"layers": [
            {"mediaType": "application/vnd.ollama.image.model", "size": size_bytes},
            {"mediaType": "application/vnd.ollama.image.license", "size": 100},
        ]})
    return httpx.MockTransport(handler)


def test_ollama_source_updates_size():
    seed = [ModelVariant(model_id="llama3.1:8b", family="Llama 3.1", params_b=8,
                         quant="Q4_K_M", file_size_gb=4.9)]
    client = httpx.Client(transport=_mock_transport_ollama(4_600_000_000))
    out = OllamaSource(seed=seed, client=client).fetch()
    assert len(out) == 1
    assert out[0].file_size_gb == 4.6
    assert out[0].source == "ollama"


def test_ollama_source_skips_on_error():
    seed = [ModelVariant(model_id="broken:tag", family="B", params_b=1, file_size_gb=1.0)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert OllamaSource(seed=seed, client=client).fetch() == []


def test_hf_source_updates_known_family_size():
    seed = [ModelVariant(model_id="qwen2.5:7b", family="Qwen2.5", params_b=7,
                         quant="Q4_K_M", file_size_gb=4.7)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"siblings": [
            {"rfilename": "qwen2.5-7b-instruct-q4_k_m.gguf", "size": 4_680_000_000},
            {"rfilename": "README.md", "size": 1000},
        ]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = HuggingFaceSource(seed=seed, client=client).fetch()
    assert len(out) == 1
    assert out[0].file_size_gb == 4.68
    assert out[0].source == "huggingface"


def test_hf_source_skips_unknown_family():
    seed = [ModelVariant(model_id="obscure:1b", family="Obscure", params_b=1, file_size_gb=1.0)]
    # No network call should be needed; unknown family is skipped.
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert HuggingFaceSource(seed=seed, client=client).fetch() == []
