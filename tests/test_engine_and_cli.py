import json

from inference_profiler import report
from inference_profiler.cli import main
from inference_profiler.engine import recommend
from inference_profiler.models import FitTier
from inference_profiler.sources import build_catalog


def test_recommend_ranks_and_tiers(rtx4090):
    cands = build_catalog(offline=True)
    rec = recommend(rtx4090, cands, context=8192, languages=["en"], use_case="chat")
    assert rec.scored
    # A 24 GB GPU fits mid-size models but not 70B Q4.
    fits = {s.variant.family for s in rec.by_tier(FitTier.FITS)}
    no_fit = {s.variant.family for s in rec.by_tier(FitTier.NO_FIT)}
    assert "Qwen2.5" in fits or "Llama 3.1" in fits
    assert "Llama 3.3" in no_fit  # 70B


def test_language_ranking_favors_multilingual(laptop_cpu):
    cands = build_catalog(offline=True)
    rec = recommend(laptop_cpu, cands, context=4096, languages=["tr", "en"])
    top_lang = rec.top_by("language", n=3)
    assert top_lang
    # Turkish support => multilingual families should top the language ranking.
    assert all(s.scores["language"] >= 50 for s in top_lang)


def test_json_output_is_valid(laptop_cpu):
    cands = build_catalog(offline=True)
    rec = recommend(laptop_cpu, cands)
    payload = json.loads(report.to_json(rec))
    assert "hardware" in payload
    assert "models" in payload
    assert payload["models"]


def test_markdown_contains_sections(laptop_cpu):
    cands = build_catalog(offline=True)
    rec = recommend(laptop_cpu, cands)
    md = report.to_markdown(rec)
    assert "# LLM Model" in md
    assert "Uygunluk" in md


def test_cli_json_smoke(capsys):
    rc = main(["--offline", "--json", "--languages", "tr,en"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["models"]


def test_cli_markdown_writes_file(tmp_path):
    out_file = tmp_path / "report.md"
    rc = main(["--offline", "--markdown", str(out_file)])
    assert rc == 0
    assert out_file.exists()
    assert "LLM Model" in out_file.read_text(encoding="utf-8")
