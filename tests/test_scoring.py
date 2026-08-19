from inference_profiler import scoring
from inference_profiler.models import FitTier, ModelVariant


def test_language_score_multi_is_full(small_variant):
    v = small_variant.model_copy(update={"languages": ["multi"]})
    assert scoring.language_score(v, ["tr", "en"]) == 100.0


def test_language_score_partial(small_variant):
    v = small_variant.model_copy(update={"languages": ["en"]})
    assert scoring.language_score(v, ["tr", "en"]) == 50.0


def test_language_score_none_wanted(small_variant):
    assert scoring.language_score(small_variant, []) == 100.0


def test_license_score_permissive_vs_nc(small_variant):
    permissive = small_variant.model_copy(update={"license": "apache-2.0"})
    nc = small_variant.model_copy(update={"license": "cc-by-nc-4.0"})
    assert scoring.license_score(permissive) > scoring.license_score(nc)


def test_context_score_monotonic():
    small = ModelVariant(model_id="a", family="A", params_b=1, file_size_gb=1, context_max=8192)
    big = ModelVariant(model_id="b", family="B", params_b=1, file_size_gb=1, context_max=131072)
    assert scoring.context_score(big) > scoring.context_score(small)


def test_weights_for_use_case():
    assert scoring.weights_for("coding")["quality"] > scoring.weights_for("chat")["quality"]
    assert scoring.weights_for(None) == scoring.DEFAULT_WEIGHTS
    assert scoring.weights_for("nonexistent") == scoring.DEFAULT_WEIGHTS


def test_best_runnable_prefers_higher_overall(laptop_cpu):
    variants = [
        ModelVariant(model_id="q4", family="X", params_b=7, quant="Q4_K_M",
                     file_size_gb=4.5, quality_score=70, languages=["multi"], license="apache-2.0"),
        ModelVariant(model_id="q8", family="X", params_b=7, quant="Q8_0",
                     file_size_gb=7.6, quality_score=72, languages=["multi"], license="apache-2.0"),
    ]
    weights = scoring.weights_for(None)
    best = scoring.best_runnable_variant(variants, laptop_cpu, 8192, ["en"], weights)
    assert best is not None
    assert best.fit.tier != FitTier.NO_FIT


def test_best_runnable_falls_back_to_smallest_when_none_fit(laptop_cpu):
    variants = [
        ModelVariant(model_id="big", family="Y", params_b=70, file_size_gb=43.0),
        ModelVariant(model_id="bigger", family="Y", params_b=70, quant="Q8_0", file_size_gb=75.0),
    ]
    weights = scoring.weights_for(None)
    best = scoring.best_runnable_variant(variants, laptop_cpu, 8192, ["en"], weights)
    assert best is not None
    assert best.fit.tier == FitTier.NO_FIT
    assert best.variant.file_size_gb == 43.0  # the smaller one
