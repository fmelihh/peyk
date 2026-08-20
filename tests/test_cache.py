
from peyk import cache
from peyk.models import ModelVariant


def _use_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


def test_cache_dir_respects_xdg(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    assert cache.cache_dir() == tmp_path / "peyk"
    assert cache.cache_dir().exists()


def test_cached_variants_hits_cache_second_time(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    calls = {"n": 0}

    def producer():
        calls["n"] += 1
        return [ModelVariant(model_id="a:7b", family="A", params_b=7, file_size_gb=4.0)]

    first = cache.cached_variants("k", 3600, producer)
    second = cache.cached_variants("k", 3600, producer)
    assert calls["n"] == 1  # second served from cache
    assert first[0].model_id == second[0].model_id == "a:7b"


def test_cache_expires(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    cache.write("k", [{"x": 1}])
    assert cache.read_fresh("k", ttl=3600) == [{"x": 1}]
    assert cache.read_fresh("k", ttl=0) is None  # stale immediately


def test_no_cache_always_calls_producer(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    calls = {"n": 0}

    def producer():
        calls["n"] += 1
        return [ModelVariant(model_id="a:7b", family="A", params_b=7, file_size_gb=4.0)]

    cache.cached_variants("k", 3600, producer, use_cache=False)
    cache.cached_variants("k", 3600, producer, use_cache=False)
    assert calls["n"] == 2


def test_key_for_is_order_independent():
    assert cache.key_for("ollama", ["b", "a"]) == cache.key_for("ollama", ["a", "b"])
