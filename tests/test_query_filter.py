import json

from peyk.cli import main
from peyk.resolve import filter_candidates
from peyk.sources import build_catalog

CATALOG = build_catalog(offline=True)


def test_filter_by_family():
    out = filter_candidates("qwen", CATALOG)
    assert out
    assert all("qwen" in c.family.lower() or
               any("qwen" in v.model_id for v in c.variants) for c in out)
    assert len(out) < len(CATALOG)


def test_filter_empty_query_returns_all():
    assert len(filter_candidates("", CATALOG)) == len(CATALOG)


def test_filter_no_match_returns_empty():
    assert filter_candidates("zzznope", CATALOG) == []


def test_cli_query_filters(capsys):
    rc = main(["qwen", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"]
    assert all("qwen" in m["model"].lower() for m in payload["models"])


def test_cli_unknown_query_falls_back_to_all(capsys):
    rc = main(["zzznope", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["models"]) > 10  # showed everything


def test_detect_baseline_cpu_model():
    from peyk.profiler import detect
    # On Linux/macOS the baseline reads a CPU model; allow None on other OSes.
    assert detect().cpu_model is None or isinstance(detect().cpu_model, str)
