import json
from datetime import date, timedelta

from peyk import report
from peyk.cli import main
from peyk.gpus import parse_gpu_arg
from peyk.sources import build_catalog


def test_catalog_meta_populated_after_build():
    build_catalog(offline=True)
    from peyk import sources
    assert sources.CATALOG_META.get("updated")
    assert sources.CATALOG_META.get("origin") == "bundled"


def test_catalog_age_days():
    recent = {"updated": (date.today() - timedelta(days=10)).isoformat()}
    assert report.catalog_age_days(recent) == 10
    assert report.catalog_age_days({}) is None
    assert report.catalog_age_days({"updated": "not-a-date"}) is None


def test_catalog_line_warns_when_stale():
    stale = {"updated": (date.today() - timedelta(days=90)).isoformat(), "origin": "bundled"}
    line = report.catalog_line(stale)
    assert line is not None
    assert "out of date" in line.plain
    fresh = {"updated": date.today().isoformat(), "origin": "bundled"}
    assert "out of date" not in report.catalog_line(fresh).plain


def test_json_includes_catalog_meta(capsys):
    rc = main(["--offline", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "catalog" in payload
    assert payload["catalog"].get("updated")


def test_gpu_inline_vram_override_is_authoritative():
    spec, count = parse_gpu_arg("T4 2GB")
    assert spec.vram_gb == 2  # overrides the DB's 16 GB T4
    spec2, _ = parse_gpu_arg("A100 40GB")
    assert spec2.vram_gb == 40
