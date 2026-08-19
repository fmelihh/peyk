import httpx
import pytest

from inference_profiler.sources.hf_discovery import (
    HuggingFaceDiscoverySource,
    estimated_quality,
    parse_params_b,
    parse_quant,
)


@pytest.mark.parametrize("repo,expected", [
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", 8.0),
    ("Qwen/Qwen2.5-7B-Instruct-GGUF", 7.0),
    ("org/SmolLM2-1.7B-Instruct-GGUF", 1.7),
    ("org/tiny-0.5B-GGUF", 0.5),
    ("org/no-size-here-GGUF", None),
])
def test_parse_params_b(repo, expected):
    assert parse_params_b(repo) == expected


@pytest.mark.parametrize("fname,expected", [
    ("model-q4_k_m.gguf", "Q4_K_M"),
    ("model.Q8_0.gguf", "Q8_0"),
    ("model-IQ4_XS.gguf", "IQ4_XS"),
    ("model-f16.gguf", "F16"),
    ("readme.md", None),
])
def test_parse_quant(fname, expected):
    assert parse_quant(fname) == expected


def test_estimated_quality_monotonic():
    assert estimated_quality(1) < estimated_quality(8) < estimated_quality(70)


def _transport(list_payload, tree_payload):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/models":
            return httpx.Response(200, json=list_payload)
        if "/tree/main" in request.url.path:
            return httpx.Response(200, json=tree_payload)
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_discovery_builds_variants_and_sums_splits():
    list_payload = [{
        "id": "someorg/NewModel-9B-Instruct-GGUF",
        "tags": ["gguf", "license:apache-2.0", "language:en", "language:tr"],
        "downloads": 99999,
    }]
    tree_payload = [
        {"type": "file", "path": "newmodel-9b-q4_k_m-00001-of-00002.gguf", "size": 3_000_000_000},
        {"type": "file", "path": "newmodel-9b-q4_k_m-00002-of-00002.gguf", "size": 2_500_000_000},
        {"type": "file", "path": "newmodel-9b-q8_0.gguf", "size": 9_500_000_000},
        {"type": "file", "path": "README.md", "size": 500},
    ]
    client = httpx.Client(transport=_transport(list_payload, tree_payload))
    out = HuggingFaceDiscoverySource(exclude_families=["Llama 3.1"], client=client).fetch()

    quants = {v.quant: v for v in out}
    assert set(quants) == {"Q4_K_M", "Q8_0"}
    assert quants["Q4_K_M"].file_size_gb == 5.5  # split files summed
    assert quants["Q4_K_M"].params_b == 9.0
    assert "tr" in quants["Q4_K_M"].languages
    assert quants["Q4_K_M"].license == "apache-2.0"
    assert quants["Q4_K_M"].source == "hf-discovered"


def test_discovery_skips_curated_families():
    list_payload = [{
        "id": "bartowski/Llama-3.1-8B-Instruct-GGUF",
        "tags": ["gguf"], "downloads": 100000,
    }]
    client = httpx.Client(transport=_transport(list_payload, []))
    # "Llama 3.1" -> keyword "llama" excludes this repo.
    out = HuggingFaceDiscoverySource(exclude_families=["Llama 3.1"], client=client).fetch()
    assert out == []


def test_discovery_skips_non_llm_by_name():
    list_payload = [{"id": "nvidia/parakeet-ctc-1.1b-GGUF", "tags": ["gguf"], "downloads": 5}]
    client = httpx.Client(transport=_transport(list_payload, []))
    assert HuggingFaceDiscoverySource(client=client).fetch() == []


def test_discovery_skips_non_llm_by_tag():
    list_payload = [{
        "id": "org/SomeModel-7B-GGUF",
        "tags": ["gguf", "automatic-speech-recognition"], "downloads": 5,
    }]
    client = httpx.Client(transport=_transport(list_payload, []))
    assert HuggingFaceDiscoverySource(client=client).fetch() == []


def test_discovery_graceful_on_list_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    assert HuggingFaceDiscoverySource(client=client).fetch() == []
