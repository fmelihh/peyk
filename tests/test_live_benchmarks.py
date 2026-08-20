import httpx
import pytest

from peyk import benchmarks
from peyk.benchmarks import live
from peyk.models import ModelVariant


@pytest.fixture(autouse=True)
def _clear_live():
    yield
    benchmarks.activate_live([])  # reset module overlay so tests don't leak


def _v(family, params):
    return ModelVariant(model_id="x", family=family, params_b=params,
                        file_size_gb=1.0, quality_score=50.0)


def test_fetch_live_disabled_without_url(monkeypatch):
    monkeypatch.delenv(live.ENV_URL, raising=False)
    assert live.fetch_live() == []


def test_fetch_live_parses_entries():
    payload = {"entries": [
        {"family": "Qwen2.5", "params_b": 7, "quality": 99, "source": "livebench"},
        {"family": "Bad"},  # missing fields -> filtered out
    ]}
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))
    out = live.fetch_live(url="https://example.com/b.json", client=client)
    assert len(out) == 1
    assert out[0]["family"] == "Qwen2.5"


def test_fetch_live_graceful_on_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert live.fetch_live(url="https://example.com/b.json", client=client) == []


def test_live_overrides_frozen():
    # Frozen Qwen2.5 7B is 73 (direct); live overlay bumps it and marks 'live'.
    before = benchmarks.evaluate(_v("Qwen2.5", 7))
    assert before.level == "direct" and before.base == 73
    benchmarks.activate_live([{"family": "Qwen2.5", "params_b": 7, "quality": 95,
                               "source": "livebench"}])
    after = benchmarks.evaluate(_v("Qwen2.5", 7))
    assert after.level == "live"
    assert after.base == 95
    assert after.effective == 95.0  # x1.0
    assert after.source == "livebench"


def test_load_live_noop_without_url(monkeypatch):
    monkeypatch.delenv(live.ENV_URL, raising=False)
    assert benchmarks.load_live() == 0
