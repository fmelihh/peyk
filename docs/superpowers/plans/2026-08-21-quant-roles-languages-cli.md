# Quant Ladder, Model Roles, Language Levels & CLI Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let peyk evaluate every quantization a machine can actually run, rank embedding and reranker models alongside LLMs, score language support from real per-language data, and tidy the flag surface — without breaking existing usage.

**Architecture:** A new `quant.py` derives a full quantization ladder from each catalog anchor (size scaled by a per-family factor measured from real catalog sizes) and filters it by what the detected accelerator supports; `engine.recommend` scores the whole ladder instead of a single catalog row. A new `languages.py` turns `ModelVariant.languages` into levelled data with an evidence discount, mirroring the existing `benchmarks.py` evidence design. A `role` field splits scoring criteria three ways and a new `rag.py` picks a co-resident embedding + reranker + LLM triple. Finally the CLI collapses its source/hardware flag clusters, keeping every old flag as a hidden deprecated alias.

**Tech Stack:** Python 3.10+, pydantic v2, rich, argparse, pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-08-21-quant-roles-languages-cli-design.md`

## Global Constraints

- **Offline-first.** Every feature must produce a full answer with no network. The quant ladder is computed, not fetched; language levels and embedding metadata are bundled in the catalog.
- **Auditable.** Every number peyk prints must be traceable to a measured input or a documented heuristic, and labelled as to which.
- **Backward compatible.** Third-party catalogs (`--catalog-url`) and HF-discovered variants use the current schema. All new fields carry defaults; all changed fields accept their old shape.
- **Report-only.** peyk never downloads, converts, quantizes, or runs a model. `snippet` only prints commands.
- Python 3.10 is the floor — no `match`, no PEP 604 in runtime positions that 3.10 rejects, no `tomllib`-only config.
- Every module starts with `from __future__ import annotations`, matching the existing codebase.
- `ruff check .` and `mypy src/peyk` must pass before every commit.

## File Structure

**Created:**
- `src/peyk/quant.py` — quantization spec table, hardware availability, anchored sizing, ladder expansion.
- `src/peyk/languages.py` — level/evidence enums, weights, and the language score function.
- `src/peyk/rag.py` — co-resident embedding + reranker + LLM stack selection.
- `docs/languages.md` — per-family language curation sources.
- `tests/test_quant.py`, `tests/test_languages.py`, `tests/test_roles.py`, `tests/test_rag.py`, `tests/test_cli_flags.py`

**Modified:**
- `src/peyk/models.py` — new enums; `ModelVariant` fields; `LanguageSupport`; `FitResult.index_gb`.
- `src/peyk/estimator.py` — `kv_bits`, role-aware KV and overhead, index cost.
- `src/peyk/scoring.py` — quant retention, levelled language score, role criteria, index-cost score.
- `src/peyk/engine.py` — ladder expansion, per-role criteria, role filter.
- `src/peyk/cli.py` — new commands and flags, deprecated aliases, argument groups.
- `src/peyk/report.py` — Lang column, derived-quant marker, `quant` / `languages` / `rag` renderers.
- `src/peyk/snippet.py` — derived-tag caveat.
- `src/peyk/config.py` — new config keys.
- `src/peyk/sources/data/catalog.json` — language migration, 16 new role entries.
- `README.md`, `docs/design.md`, `docs/roadmap.md`, `examples/`

---

# Phase 1 — Quantization ladder

### Task 1: Data-model foundations

**Files:**
- Modify: `src/peyk/models.py:81-102`
- Test: `tests/test_quant.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Role`, `QuantFormat`, `SizeEvidence` enums; `ModelVariant.role`, `.quant_format`, `.size_evidence`, `.quant_tag`, `.embed_dim`; `FitResult.index_gb`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quant.py
from peyk.models import ModelVariant, QuantFormat, Role, SizeEvidence


def test_variant_defaults_are_backward_compatible():
    v = ModelVariant(model_id="x:7b", family="X", params_b=7, file_size_gb=4.1)
    assert v.role is Role.LLM
    assert v.quant_format is QuantFormat.GGUF
    assert v.size_evidence is SizeEvidence.MEASURED
    assert v.quant_tag is None
    assert v.embed_dim is None


def test_variant_accepts_new_fields():
    v = ModelVariant(
        model_id="bge-m3", family="BGE-M3", params_b=0.568, file_size_gb=1.2,
        role="embedding", embed_dim=1024, quant_format="gguf",
        size_evidence="derived", quant_tag="q8_0",
    )
    assert v.role is Role.EMBEDDING
    assert v.embed_dim == 1024
    assert v.size_evidence is SizeEvidence.DERIVED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quant.py -v`
Expected: FAIL — `ImportError: cannot import name 'Role' from 'peyk.models'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/peyk/models.py`, after the `Accelerator` enum:

```python
class Role(str, Enum):
    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


class QuantFormat(str, Enum):
    GGUF = "gguf"
    AWQ = "awq"
    GPTQ = "gptq"
    FP8 = "fp8"
    NF4 = "nf4"
    MLX = "mlx"
    FP16 = "fp16"


class SizeEvidence(str, Enum):
    MEASURED = "measured"   # a real file size from the catalog or a registry
    DERIVED = "derived"     # computed from the anchored bits-per-weight model
```

Add to `ModelVariant`, after `gqa_factor`:

```python
    role: Role = Role.LLM
    quant_format: QuantFormat = QuantFormat.GGUF
    size_evidence: SizeEvidence = SizeEvidence.MEASURED
    quant_tag: str | None = None   # conventional registry suffix, e.g. "q6_K"
    embed_dim: int | None = None   # embedding role only: output vector width
```

Add to `FitResult`, after `overhead_gb`:

```python
    index_gb: float = 0.0  # vector-index cost (embedding role); 0 elsewhere
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_quant.py -v && pytest -q && ruff check . && mypy src/peyk`
Expected: new tests PASS, full suite still green.

- [ ] **Step 5: Commit**

```bash
git add src/peyk/models.py tests/test_quant.py
git commit -m "Add role, quant-format and size-evidence fields to ModelVariant"
```

---

### Task 2: Quantization spec table and anchored sizing

**Files:**
- Create: `src/peyk/quant.py`
- Test: `tests/test_quant.py`

**Interfaces:**
- Consumes: `Role`, `QuantFormat` from Task 1.
- Produces: `QuantSpec` dataclass (`name`, `bits_per_weight`, `quality_retention`, `format`, `accelerators`, `min_compute_cap`, `tag`); `QUANTS: dict[str, QuantSpec]`; `spec(name) -> QuantSpec | None`; `retention(quant: str) -> float`; `anchor_factor(variants) -> float`; `derived_size_gb(params_b, q, k) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quant.py  (append)
import pytest

from peyk import quant


def test_known_quant_specs():
    assert quant.spec("Q4_K_M").bits_per_weight == pytest.approx(4.83)
    assert quant.spec("q4_k_m").name == "Q4_K_M"       # lookup is case-insensitive
    assert quant.spec("F16").quality_retention == 1.0
    assert quant.spec("NOT_A_QUANT") is None


def test_retention_defaults_to_neutral_for_unknown_quants():
    # HF-discovered variants carry quant names we do not model; they must not be
    # penalised by an accidental 0.0 multiplier.
    assert quant.retention("Q4_K_M") == pytest.approx(0.985)
    assert quant.retention("IQ4_NL_CUSTOM") == 1.0


def test_retention_is_monotonic_in_bits():
    gguf = [q for q in quant.QUANTS.values() if q.format is quant.QuantFormat.GGUF]
    ordered = sorted(gguf, key=lambda q: q.bits_per_weight)
    retentions = [q.quality_retention for q in ordered]
    assert retentions == sorted(retentions)


def test_anchor_factor_absorbs_per_family_overhead(small_variant):
    # llama3.1:8b is 4.9 GB at Q4_K_M; raw bpw math predicts 8 * 4.83/8 = 4.83.
    v = small_variant.model_copy(update={"params_b": 8.0, "file_size_gb": 4.9})
    k = quant.anchor_factor([v])
    assert k == pytest.approx(4.9 / 4.83, rel=1e-3)


def test_anchor_predicts_the_measured_q8_size():
    # The catalog holds both llama3.1:8b Q4_K_M (4.9 GB) and Q8_0 (8.5 GB).
    # Anchoring on Q4 must predict Q8 within 3%.
    from peyk.models import ModelVariant
    q4 = ModelVariant(model_id="llama3.1:8b", family="Llama 3.1", params_b=8,
                      quant="Q4_K_M", file_size_gb=4.9)
    k = quant.anchor_factor([q4])
    predicted = quant.derived_size_gb(8.0, quant.spec("Q8_0"), k)
    assert predicted == pytest.approx(8.5, rel=0.03)


def test_anchor_factor_ignores_unmodelled_quants():
    from peyk.models import ModelVariant
    odd = ModelVariant(model_id="x:7b", family="X", params_b=7,
                       quant="IQ4_NL_CUSTOM", file_size_gb=4.0)
    assert quant.anchor_factor([odd]) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quant.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'peyk.quant'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/peyk/quant.py
"""Quantization ladder: what each quant costs, what it preserves, where it runs.

Two tables drive everything here.

`bits_per_weight` are llama.cpp *effective* bpw — they include the
`token_embd`/`output` tensor overrides, which is why Q4_K_M is 4.83 and not the
nominal 4.5. Non-GGUF formats use their published weight widths.

`quality_retention` is a multiplier on the model's benchmark quality, derived
from published perplexity deltas on Llama-2-7B. It is expressed as a multiplier
so it composes with `benchmarks.evaluate()`. The bundled benchmark snapshot is
treated as measured at ~fp16; see docs/design.md.

Both tables are documented heuristics, not measurements of your file. Sizes,
however, are *anchored* to real catalog file sizes (see `anchor_factor`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .models import Accelerator, HardwareProfile, ModelVariant, QuantFormat

# Compute capability assumed for an NVIDIA GPU whose capability is unknown
# (simulated cards, driverless hosts). Deliberately below FP8's 8.9 floor: we
# omit FP8 rather than promise it.
ASSUMED_COMPUTE_CAP = 7.5

_ANY = None  # accelerators=None -> runs anywhere, CPU included


@dataclass(frozen=True)
class QuantSpec:
    name: str
    bits_per_weight: float
    quality_retention: float
    format: QuantFormat
    tag: str | None = None                      # registry suffix, e.g. "q6_K"
    accelerators: frozenset[Accelerator] | None = _ANY
    min_compute_cap: float | None = None


def _g(name: str, bpw: float, ret: float, tag: str) -> QuantSpec:
    return QuantSpec(name, bpw, ret, QuantFormat.GGUF, tag)


_SPECS: list[QuantSpec] = [
    # GGUF k-quants — llama.cpp / Ollama / LM Studio, CPU and GPU alike.
    _g("F16", 16.0, 1.000, "f16"),
    _g("Q8_0", 8.5, 0.999, "q8_0"),
    _g("Q6_K", 6.56, 0.997, "q6_K"),
    _g("Q5_K_M", 5.67, 0.993, "q5_K_M"),
    _g("Q5_K_S", 5.52, 0.991, "q5_K_S"),
    _g("Q4_K_M", 4.83, 0.985, "q4_K_M"),
    _g("Q4_K_S", 4.57, 0.981, "q4_K_S"),
    _g("IQ4_XS", 4.25, 0.978, "iq4_xs"),
    _g("Q3_K_M", 3.89, 0.955, "q3_K_M"),
    _g("IQ3_M", 3.66, 0.945, "iq3_m"),
    _g("Q3_K_S", 3.50, 0.930, "q3_K_S"),
    _g("IQ2_M", 2.70, 0.900, "iq2_m"),
    _g("Q2_K", 2.63, 0.880, "q2_K"),
    # GPU-only formats (vLLM / transformers / TensorRT-LLM / MLX).
    QuantSpec("FP16", 16.0, 1.000, QuantFormat.FP16, None,
              frozenset({Accelerator.NVIDIA, Accelerator.AMD})),
    QuantSpec("FP8", 8.0, 0.998, QuantFormat.FP8, None,
              frozenset({Accelerator.NVIDIA}), 8.9),
    QuantSpec("AWQ-4", 4.15, 0.982, QuantFormat.AWQ, None,
              frozenset({Accelerator.NVIDIA}), 7.5),
    QuantSpec("GPTQ-4", 4.15, 0.978, QuantFormat.GPTQ, None,
              frozenset({Accelerator.NVIDIA, Accelerator.AMD}), 6.1),
    QuantSpec("NF4", 4.50, 0.975, QuantFormat.NF4, None,
              frozenset({Accelerator.NVIDIA})),
    QuantSpec("MLX-8", 8.5, 0.999, QuantFormat.MLX, None,
              frozenset({Accelerator.APPLE})),
    QuantSpec("MLX-4", 4.50, 0.980, QuantFormat.MLX, None,
              frozenset({Accelerator.APPLE})),
]

QUANTS: dict[str, QuantSpec] = {q.name: q for q in _SPECS}
_BY_LOWER: dict[str, QuantSpec] = {q.name.lower(): q for q in _SPECS}


def spec(name: str) -> QuantSpec | None:
    return _BY_LOWER.get(name.strip().lower())


def retention(quant: str) -> float:
    """Quality multiplier for a quant. Unknown quants are not penalised."""
    q = spec(quant)
    return q.quality_retention if q else 1.0


def raw_size_gb(params_b: float, q: QuantSpec) -> float:
    """Unanchored size: parameters x bits / 8, in GB."""
    return params_b * q.bits_per_weight / 8.0


def anchor_factor(variants: Sequence[ModelVariant]) -> float:
    """Per-family size correction measured from real catalog file sizes.

    Raw bpw math ignores vocabulary/embedding overhead, which varies by family.
    Dividing each measured size by its predicted size recovers that factor;
    averaging over every measured row of the family keeps it stable. Returns
    1.0 when no variant uses a quant we model.
    """
    factors = []
    for v in variants:
        q = spec(v.quant)
        if q is None or v.params_b <= 0 or v.file_size_gb <= 0:
            continue
        predicted = raw_size_gb(v.params_b, q)
        if predicted > 0:
            factors.append(v.file_size_gb / predicted)
    return sum(factors) / len(factors) if factors else 1.0


def derived_size_gb(params_b: float, q: QuantSpec, k: float) -> float:
    return round(raw_size_gb(params_b, q) * k, 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_quant.py -v && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/quant.py tests/test_quant.py
git commit -m "Add quantization spec table with anchored size derivation"
```

---

### Task 3: Hardware availability filter

**Files:**
- Modify: `src/peyk/quant.py`
- Test: `tests/test_quant.py`

**Interfaces:**
- Consumes: `QuantSpec`, `QUANTS` from Task 2.
- Produces: `available_for(hw: HardwareProfile) -> list[QuantSpec]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quant.py  (append)
from peyk.models import Accelerator, QuantFormat


def _formats(specs):
    return {q.format for q in specs}


def test_cpu_only_gets_gguf_only(laptop_cpu):
    assert _formats(quant.available_for(laptop_cpu)) == {QuantFormat.GGUF}


def test_apple_gets_gguf_and_mlx(mac_m3):
    assert _formats(quant.available_for(mac_m3)) == {QuantFormat.GGUF, QuantFormat.MLX}


def test_nvidia_without_known_capability_omits_fp8(rtx4090):
    names = {q.name for q in quant.available_for(rtx4090)}
    assert "AWQ-4" in names and "GPTQ-4" in names and "NF4" in names
    assert "FP8" not in names          # capability unknown -> not promised


def test_nvidia_ada_gets_fp8(rtx4090):
    ada = rtx4090.model_copy(update={"gpu_compute_cap": "8.9"})
    assert "FP8" in {q.name for q in quant.available_for(ada)}


def test_nvidia_pascal_loses_awq(rtx4090):
    pascal = rtx4090.model_copy(update={"gpu_compute_cap": "6.1"})
    names = {q.name for q in quant.available_for(pascal)}
    assert "GPTQ-4" in names
    assert "AWQ-4" not in names


def test_amd_gets_gguf_gptq_fp16(laptop_cpu):
    amd = laptop_cpu.model_copy(update={"accelerator": Accelerator.AMD,
                                        "vram_total_gb": 24.0})
    assert _formats(quant.available_for(amd)) == {
        QuantFormat.GGUF, QuantFormat.GPTQ, QuantFormat.FP16}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quant.py -k available -v`
Expected: FAIL — `AttributeError: module 'peyk.quant' has no attribute 'available_for'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/peyk/quant.py`:

```python
def _compute_cap(hw: HardwareProfile) -> float:
    """Numeric compute capability, or a conservative assumption when unknown."""
    if hw.gpu_compute_cap:
        try:
            return float(hw.gpu_compute_cap)
        except ValueError:
            pass
    return ASSUMED_COMPUTE_CAP


def available_for(hw: HardwareProfile) -> list[QuantSpec]:
    """Quantizations this machine can actually load, in bpw order (high first).

    GGUF runs everywhere including CPU. GPU formats are gated on the vendor and,
    for FP8/AWQ, on compute capability. When capability is unknown we assume the
    conservative baseline in ASSUMED_COMPUTE_CAP, which excludes FP8.
    """
    cap = _compute_cap(hw)
    out = []
    for q in _SPECS:
        if q.accelerators is not None:
            if hw.accelerator not in q.accelerators:
                continue
            if hw.accelerator is Accelerator.NVIDIA and q.min_compute_cap:
                if cap < q.min_compute_cap:
                    continue
        out.append(q)
    return sorted(out, key=lambda q: q.bits_per_weight, reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_quant.py -v && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/quant.py tests/test_quant.py
git commit -m "Gate quantization formats on accelerator and compute capability"
```

---

### Task 4: Ladder expansion

**Files:**
- Modify: `src/peyk/quant.py`
- Test: `tests/test_quant.py`

**Interfaces:**
- Consumes: `available_for`, `anchor_factor`, `derived_size_gb`.
- Produces: `expand(cand: ModelCandidate, hw: HardwareProfile, *, only_quant: str | None = None, only_format: str | None = None) -> list[ModelVariant]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quant.py  (append)
from peyk.models import ModelCandidate, SizeEvidence


def _llama8b() -> ModelCandidate:
    return ModelCandidate(family="Llama 3.1", params_b=8, variants=[
        ModelVariant(model_id="llama3.1:8b", family="Llama 3.1", params_b=8,
                     quant="Q4_K_M", file_size_gb=4.9, context_max=131072),
        ModelVariant(model_id="llama3.1:8b-q8", family="Llama 3.1", params_b=8,
                     quant="Q8_0", file_size_gb=8.5, context_max=131072),
    ])


def test_expand_keeps_measured_rows_and_adds_derived(laptop_cpu):
    out = quant.expand(_llama8b(), laptop_cpu)
    by_quant = {v.quant: v for v in out}
    assert by_quant["Q4_K_M"].size_evidence is SizeEvidence.MEASURED
    assert by_quant["Q4_K_M"].file_size_gb == 4.9      # untouched
    assert by_quant["Q8_0"].size_evidence is SizeEvidence.MEASURED
    assert by_quant["Q6_K"].size_evidence is SizeEvidence.DERIVED
    assert 6.0 < by_quant["Q6_K"].file_size_gb < 7.5


def test_expand_carries_anchor_metadata(laptop_cpu):
    out = quant.expand(_llama8b(), laptop_cpu)
    derived = next(v for v in out if v.quant == "Q6_K")
    assert derived.family == "Llama 3.1"
    assert derived.context_max == 131072
    assert derived.quant_tag == "q6_K"
    assert derived.model_id == "llama3.1:8b"   # base id; tag is separate


def test_expand_offers_no_gpu_formats_on_cpu(laptop_cpu):
    assert all(v.quant_format is QuantFormat.GGUF
               for v in quant.expand(_llama8b(), laptop_cpu))


def test_expand_filters_to_a_single_quant(laptop_cpu):
    out = quant.expand(_llama8b(), laptop_cpu, only_quant="Q6_K")
    assert [v.quant for v in out] == ["Q6_K"]


def test_expand_filters_by_format(mac_m3):
    out = quant.expand(_llama8b(), mac_m3, only_format="mlx")
    assert out and all(v.quant_format is QuantFormat.MLX for v in out)


def test_expand_passes_through_unmodelled_variants(laptop_cpu):
    odd = ModelCandidate(family="Weird", params_b=7, variants=[
        ModelVariant(model_id="weird:7b", family="Weird", params_b=7,
                     quant="IQ4_NL_CUSTOM", file_size_gb=4.0,
                     source="hf-discovered"),
    ])
    out = quant.expand(odd, laptop_cpu)
    assert [v.quant for v in out] == ["IQ4_NL_CUSTOM"]
    assert out[0].size_evidence is SizeEvidence.MEASURED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quant.py -k expand -v`
Expected: FAIL — `AttributeError: module 'peyk.quant' has no attribute 'expand'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/peyk/quant.py` (and add `ModelCandidate`, `SizeEvidence` to the
`.models` import at the top):

```python
def expand(
    cand: ModelCandidate,
    hw: HardwareProfile,
    *,
    only_quant: str | None = None,
    only_format: str | None = None,
) -> list[ModelVariant]:
    """Every quantization of this model the machine could run.

    Catalog rows keep their measured file size; the rest are derived from the
    anchored bpw model. A candidate whose quants we do not model (typically an
    HF-discovered repackage) passes through untouched — we will not invent a
    ladder from a size we cannot interpret.
    """
    measured = {v.quant.lower(): v for v in cand.variants if spec(v.quant)}
    if not measured:
        return list(cand.variants)

    anchor = max(measured.values(), key=lambda v: v.file_size_gb)
    k = anchor_factor(list(measured.values()))

    out: list[ModelVariant] = []
    for q in available_for(hw):
        if only_quant and q.name.lower() != only_quant.strip().lower():
            continue
        if only_format and q.format.value != only_format.strip().lower():
            continue
        hit = measured.get(q.name.lower())
        if hit is not None:
            out.append(hit)
            continue
        out.append(anchor.model_copy(update={
            "quant": q.name,
            "quant_format": q.format,
            "quant_tag": q.tag,
            "file_size_gb": derived_size_gb(anchor.params_b, q, k),
            "size_evidence": SizeEvidence.DERIVED,
        }))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_quant.py -v && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/quant.py tests/test_quant.py
git commit -m "Expand each catalog model into the quantizations its host can run"
```

---

### Task 5: KV-cache quantization

**Files:**
- Modify: `src/peyk/estimator.py:27-42`, `src/peyk/estimator.py:97-111`
- Test: `tests/test_estimator.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `kv_cache_gb(context, variant, kv_bits: int = 16)`, `memory_need_gb(variant, context, kv_bits: int = 16)`, `estimate_fit(variant, hw, context, kv_bits: int = 16)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_estimator.py  (append)
from peyk.estimator import estimate_fit, kv_cache_gb


def test_kv_cache_halves_at_8_bit(small_variant):
    full = kv_cache_gb(32768, small_variant)
    half = kv_cache_gb(32768, small_variant, kv_bits=8)
    quarter = kv_cache_gb(32768, small_variant, kv_bits=4)
    assert half == pytest.approx(full / 2)
    assert quarter == pytest.approx(full / 4)


def test_kv_quant_can_rescue_a_tight_fit(big_variant, rtx4090):
    ctx = 131072
    full = estimate_fit(big_variant, rtx4090, ctx)
    lean = estimate_fit(big_variant, rtx4090, ctx, kv_bits=4)
    assert lean.mem_need_gb < full.mem_need_gb
    assert lean.kv_cache_gb == pytest.approx(full.kv_cache_gb / 4, rel=1e-2)
```

Add `import pytest` at the top of the file if it is not already there.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_estimator.py -k kv -v`
Expected: FAIL — `TypeError: kv_cache_gb() got an unexpected keyword argument 'kv_bits'`

- [ ] **Step 3: Write minimal implementation**

In `src/peyk/estimator.py`, change the three functions:

```python
def kv_cache_gb(context: int, variant: ModelVariant, kv_bits: int = 16) -> float:
    """Estimate KV-cache size for a full context window.

    Prefers real architecture numbers when the catalog provides them, otherwise
    falls back to a params-scaled, GQA-aware approximation. `kv_bits` models
    KV-cache quantization (llama.cpp `-ctk`/`-ctv`): the cache scales linearly
    with its element width, so q8_0 halves it and q4_0 quarters it.
    """
    scale = kv_bits / 16.0
    if variant.n_layers and variant.hidden:
        bytes_total = 2 * variant.n_layers * variant.hidden * 2 * context
        return bytes_total / 1e9 * variant.gqa_factor * scale
    return context * variant.params_b * KV_PER_TOKEN_PARAM * variant.gqa_factor * scale


def memory_need_gb(variant: ModelVariant, context: int, kv_bits: int = 16) -> float:
    """Total memory to load weights + KV cache + runtime overhead."""
    return variant.file_size_gb + kv_cache_gb(context, variant, kv_bits) + RUNTIME_OVERHEAD_GB


def estimate_fit(
    variant: ModelVariant, hw: HardwareProfile, context: int, kv_bits: int = 16
) -> FitResult:
    ctx = min(context, variant.context_max)
    kv = kv_cache_gb(ctx, variant, kv_bits)
    mem_need = variant.file_size_gb + kv + RUNTIME_OVERHEAD_GB
    tier = classify(mem_need, hw.memory_pool_gb)
    tps = estimate_tokens_per_sec(variant, hw, _effective_bandwidth(mem_need, hw))
    return FitResult(
        variant=variant,
        mem_need_gb=round(mem_need, 2),
        tier=tier,
        est_tokens_per_sec=round(tps, 1),
        weights_gb=round(variant.file_size_gb, 2),
        kv_cache_gb=round(kv, 2),
        overhead_gb=RUNTIME_OVERHEAD_GB,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_estimator.py -v && pytest -q && ruff check . && mypy src/peyk`
Expected: PASS — existing estimator tests are unaffected because `kv_bits` defaults to 16.

- [ ] **Step 5: Commit**

```bash
git add src/peyk/estimator.py tests/test_estimator.py
git commit -m "Model KV-cache quantization in the memory estimate"
```

---

### Task 6: Retention in scoring, ladder in the engine

**Files:**
- Modify: `src/peyk/scoring.py:46-49`, `src/peyk/scoring.py:81-128`, `src/peyk/engine.py:32-56`
- Test: `tests/test_scoring.py`, `tests/test_engine_and_cli.py`

**Interfaces:**
- Consumes: `quant.retention`, `quant.expand`.
- Produces: `score_variant(..., kv_bits: int = 16)`; `best_runnable_variant(..., kv_bits: int = 16)`; `recommend(..., kv_bits: int = 16, only_quant: str | None = None, only_format: str | None = None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring.py  (append)
from peyk.models import ModelCandidate, ModelVariant
from peyk.engine import recommend
from peyk.scoring import score_variant, weights_for


def test_quantization_loss_discounts_quality(laptop_cpu):
    base = ModelVariant(model_id="llama3.1:8b", family="Llama 3.1", params_b=8,
                        quant="Q8_0", file_size_gb=8.5)
    lossy = base.model_copy(update={"quant": "Q2_K", "file_size_gb": 2.8})
    w = weights_for(None)
    hi = score_variant(base, laptop_cpu, 8192, ["en"], w)
    lo = score_variant(lossy, laptop_cpu, 8192, ["en"], w)
    assert lo.scores["quality"] < hi.scores["quality"]


def test_recommend_upgrades_the_quant_when_memory_allows(mac_m3):
    cand = ModelCandidate(family="Llama 3.1", params_b=8, variants=[
        ModelVariant(model_id="llama3.1:8b", family="Llama 3.1", params_b=8,
                     quant="Q4_K_M", file_size_gb=4.9, context_max=131072),
    ])
    rec = recommend(hw=mac_m3, candidates=[cand], context=8192)
    chosen = rec.scored[0].variant
    # 36 GB of unified memory: a higher-fidelity quant fits and should win.
    assert chosen.quant != "Q4_K_M"
    assert quant.spec(chosen.quant).bits_per_weight > 4.83


def test_recommend_honours_an_explicit_quant(mac_m3):
    cand = ModelCandidate(family="Llama 3.1", params_b=8, variants=[
        ModelVariant(model_id="llama3.1:8b", family="Llama 3.1", params_b=8,
                     quant="Q4_K_M", file_size_gb=4.9, context_max=131072),
    ])
    rec = recommend(hw=mac_m3, candidates=[cand], context=8192, only_quant="Q4_K_M")
    assert rec.scored[0].variant.quant == "Q4_K_M"


def test_recommend_surfaces_a_model_that_only_fits_when_squeezed(laptop_cpu):
    # 16 GB CPU box, 14B model: Q4_K_M does not fit, a low-bit quant does.
    cand = ModelCandidate(family="Qwen2.5", params_b=14, variants=[
        ModelVariant(model_id="qwen2.5:14b", family="Qwen2.5", params_b=14,
                     quant="Q4_K_M", file_size_gb=9.0, context_max=32768),
    ])
    rec = recommend(hw=laptop_cpu.model_copy(update={"ram_total_gb": 8.0,
                                                     "ram_available_gb": 6.0}),
                    candidates=[cand], context=8192)
    from peyk.models import FitTier
    assert rec.scored[0].fit.tier is not FitTier.NO_FIT
```

Add `from peyk import quant` to the imports of `tests/test_scoring.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL — `test_recommend_upgrades_the_quant_when_memory_allows` asserts `chosen.quant != "Q4_K_M"` but the engine still returns the single catalog row; `recommend()` also rejects `only_quant`.

- [ ] **Step 3: Write minimal implementation**

In `src/peyk/scoring.py`, add `from . import benchmarks, quant` at the top, then:

```python
def quality_score(variant: ModelVariant) -> float:
    """Evidence-based quality, discounted by quantization loss."""
    return _clamp(benchmarks.evaluate(variant).effective * quant.retention(variant.quant))
```

and thread `kv_bits` through:

```python
def score_variant(
    variant: ModelVariant,
    hw: HardwareProfile,
    context: int,
    languages: list[str],
    weights: dict[str, float],
    kv_bits: int = 16,
) -> ScoredModel:
    fit = estimate_fit(variant, hw, context, kv_bits)
    evidence = benchmarks.evaluate(variant)
    scores = {
        "speed": round(speed_score(fit), 1),
        "quality": round(_clamp(evidence.effective * quant.retention(variant.quant)), 1),
        "language": round(language_score(variant, languages), 1),
        "context": round(context_score(variant), 1),
        "license": round(license_score(variant), 1),
    }
    overall = round(sum(scores[k] * weights.get(k, 0) for k in scores), 1)
    return ScoredModel(
        variant=variant, fit=fit, scores=scores, overall=overall,
        quality_evidence=evidence.level, quality_source=evidence.source,
    )
```

`best_runnable_variant` gains `kv_bits: int = 16` and passes it to `score_variant`.

In `src/peyk/engine.py`, add `from . import quant` and replace the variant
selection inside `recommend`:

```python
def recommend(
    hw: HardwareProfile,
    candidates: list[ModelCandidate],
    context: int = 8192,
    languages: list[str] | None = None,
    use_case: str | None = None,
    min_tps: float = 0.0,
    kv_bits: int = 16,
    only_quant: str | None = None,
    only_format: str | None = None,
) -> Recommendation:
    languages = languages or ["en"]
    weights = weights_for(use_case)
    vision_only = use_case == "vision"
    scored: list[ScoredModel] = []
    for cand in candidates:
        variants = quant.expand(cand, hw, only_quant=only_quant, only_format=only_format)
        if vision_only:
            variants = [v for v in variants if v.modality == "vision"]
        if not variants:
            continue
        best = best_runnable_variant(variants, hw, context, languages, weights, kv_bits)
        if best is None:
            continue
        if min_tps > 0 and best.fit.est_tokens_per_sec < min_tps:
            continue
        scored.append(best)
    return Recommendation(hw=hw, scored=scored, context=context)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: new tests PASS. Some existing assertions in `tests/test_engine_and_cli.py` and `tests/test_scoring.py` that pin a specific recommended quant will now fail — **update them to the new intent** (the ladder is expected to pick a higher-fidelity quant when memory allows) rather than weakening them. Note each such update in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/peyk/scoring.py src/peyk/engine.py tests/
git commit -m "Rank across the full quantization ladder, discounted by quant loss

Existing tests that pinned Q4_K_M as the recommended quant are updated:
the engine now legitimately prefers a higher-fidelity quant when the
memory pool has room for it."
```

---

### Task 7: `peyk quant` command and report wiring

**Files:**
- Modify: `src/peyk/report.py:94-133`, `src/peyk/cli.py`
- Test: `tests/test_engine_and_cli.py`

**Interfaces:**
- Consumes: `quant.expand`, `quant.spec`, `estimate_fit`.
- Produces: `report.render_quant_ladder(rows, variant_name, console)`, `report.quant_ladder_to_json(rows)`, CLI `quant` subcommand, `--quant` / `--format` / `--kv-quant` on `recommend`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_and_cli.py  (append)
def test_quant_command_lists_the_ladder(capsys):
    from peyk.cli import main
    assert main(["quant", "llama 3.1 8b", "--json"]) == 0
    import json
    data = json.loads(capsys.readouterr().out)
    quants = [r["quant"] for r in data["ladder"]]
    assert "Q4_K_M" in quants and "Q8_0" in quants
    assert len(quants) > 3
    row = next(r for r in data["ladder"] if r["quant"] == "Q4_K_M")
    assert row["size_evidence"] == "measured"
    assert 0.0 < row["quality_retention"] <= 1.0
    assert row["tier"] in {"FITS", "TIGHT", "NO_FIT"}


def test_quant_command_rejects_unknown_model(capsys):
    from peyk.cli import main
    assert main(["quant", "definitely-not-a-model"]) == 1


def test_recommend_accepts_kv_quant(capsys):
    from peyk.cli import main
    assert main(["recommend", "--kv-quant", "q4_0", "--json"]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine_and_cli.py -k quant -v`
Expected: FAIL — argparse exits 2 with "invalid choice: 'quant'"

- [ ] **Step 3: Write minimal implementation**

In `src/peyk/report.py`, mark derived quants in `_tier_table` by replacing the
quant cell:

```python
                v.quant + ("~" if v.size_evidence is SizeEvidence.DERIVED else ""),
```

(import `SizeEvidence` from `.models`), and extend `_tier_legend()` with
`"~ = size derived from the bits-per-weight model, not a measured file"`.

Add the ladder renderer:

```python
def render_quant_ladder(rows: list[dict], title: str, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(box=ROUNDED, border_style=theme.MUTED,
                  header_style=f"bold {theme.ACCENT}",
                  title=f"Quantization ladder · {title}",
                  title_style=f"bold {theme.CYAN}", title_justify="left")
    for col, justify in (("Quant", "left"), ("Format", "left"), ("Size", "right"),
                         ("Memory", "right"), ("Speed t/s", "right"),
                         ("ΔQuality", "right"), ("Status", "left")):
        table.add_column(col, justify=justify, no_wrap=True)
    for r in rows:
        delta = (r["quality_retention"] - 1.0) * 100.0
        table.add_row(
            r["quant"] + ("~" if r["size_evidence"] == "derived" else ""),
            r["format"],
            f"{r['file_size_gb']:.1f} GB",
            f"{r['mem_need_gb']:.1f} GB",
            Text(f"~{r['est_tokens_per_sec']:.0f}",
                 style=theme.speed_color(r["est_tokens_per_sec"])),
            "—" if delta == 0 else f"{delta:.1f}%",
            _tier_label(FitTier(r["tier"])),
        )
    console.print(table)
    console.print(Text("~ = size derived from the bits-per-weight model, "
                       "not a measured file", style=theme.MUTED))


def quant_ladder_to_json(rows: list[dict], title: str) -> str:
    import json
    return json.dumps({"model": title, "ladder": rows}, indent=2, ensure_ascii=False)
```

In `src/peyk/cli.py`: add `"quant"` to `SUBCOMMANDS`, add the three new flags to
`rec`, and register the subparser and handler.

```python
    rec.add_argument("--quant", metavar="NAME",
                     help="Force a quantization (default: pick the best that fits)")
    rec.add_argument("--format", metavar="FMT", dest="quant_format",
                     help="Restrict to a quant format: gguf, awq, gptq, fp8, nf4, mlx, fp16")
    rec.add_argument("--kv-quant", choices=["f16", "q8_0", "q4_0"], default="f16",
                     help="KV-cache precision (q8_0 halves it, q4_0 quarters it)")
```

```python
    qz = sub.add_parser("quant", help="Compare every quantization of one model")
    qz.add_argument("model", help='Model to compare, e.g. "qwen2.5 14b"')
    qz.add_argument("--context", type=int, default=8192, help="Target context length in tokens")
    qz.add_argument("--format", metavar="FMT", dest="quant_format",
                    help="Restrict to a quant format")
    qz.add_argument("--kv-quant", choices=["f16", "q8_0", "q4_0"], default="f16",
                    help="KV-cache precision")
    qz.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    _add_hardware_flags(qz)
    _add_common_flags(qz)
```

```python
KV_BITS = {"f16": 16, "q8_0": 8, "q4_0": 4}


def _cmd_quant(args, console: Console, status: Console) -> int:
    from . import quant as quant_mod
    from .estimator import estimate_fit
    from .resolve import resolve_variant
    candidates = build_catalog(offline=True)
    variant = resolve_variant(args.model, candidates)
    if variant is None:
        status.print(f"[red]No catalog model matched '{args.model}'.[/red]")
        return 1
    cand = next(c for c in candidates
                if c.family == variant.family and c.params_b == variant.params_b)
    hw = _resolve_hardware(args, status)
    kv_bits = KV_BITS[args.kv_quant]
    rows = []
    for v in quant_mod.expand(cand, hw, only_format=args.quant_format):
        fit = estimate_fit(v, hw, args.context, kv_bits)
        rows.append({
            "quant": v.quant,
            "format": v.quant_format.value,
            "file_size_gb": round(v.file_size_gb, 2),
            "mem_need_gb": fit.mem_need_gb,
            "est_tokens_per_sec": fit.est_tokens_per_sec,
            "quality_retention": quant_mod.retention(v.quant),
            "size_evidence": v.size_evidence.value,
            "tier": fit.tier.value,
        })
    title = f"{variant.family} {variant.params_b:g}B"
    if args.json:
        print(report.quant_ladder_to_json(rows, title))
    else:
        report.render_quant_ladder(rows, title, console=console)
    return 0
```

Register `"quant": _cmd_quant` in `_DISPATCH`, and pass the new options through
`_cmd_recommend`:

```python
    rec = recommend(hw=hw, candidates=candidates, context=args.context,
                    languages=languages, use_case=args.use_case, min_tps=min_tps,
                    kv_bits=KV_BITS[args.kv_quant], only_quant=args.quant,
                    only_format=args.quant_format)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/cli.py src/peyk/report.py tests/test_engine_and_cli.py
git commit -m "Add peyk quant, --quant/--format/--kv-quant, and a derived-size marker"
```

---

### Task 8: Honest snippets for derived quants

**Files:**
- Modify: `src/peyk/snippet.py`
- Test: `tests/test_plan_resolve_snippet.py`

**Interfaces:**
- Consumes: `ModelVariant.quant_tag`, `.size_evidence`.
- Produces: `build_snippets(variant, hw)` returning an extra `"caveat": str | None` key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plan_resolve_snippet.py  (append)
from peyk.models import ModelVariant, SizeEvidence
from peyk.snippet import build_snippets


def test_derived_quant_snippet_warns_about_the_registry_tag(laptop_cpu):
    v = ModelVariant(model_id="llama3.1:8b", family="Llama 3.1", params_b=8,
                     quant="Q6_K", file_size_gb=6.6, quant_tag="q6_K",
                     size_evidence=SizeEvidence.DERIVED)
    data = build_snippets(v, laptop_cpu)
    assert data["caveat"] is not None
    assert "q6_K" in data["ollama"]


def test_measured_quant_snippet_has_no_caveat(laptop_cpu):
    v = ModelVariant(model_id="llama3.1:8b", family="Llama 3.1", params_b=8,
                     quant="Q4_K_M", file_size_gb=4.9)
    assert build_snippets(v, laptop_cpu)["caveat"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plan_resolve_snippet.py -k caveat -v`
Expected: FAIL — `KeyError: 'caveat'`

- [ ] **Step 3: Write minimal implementation**

In `src/peyk/snippet.py`, compose the tagged id and set the caveat:

```python
    tagged = (f"{variant.model_id}-{variant.quant_tag}"
              if variant.quant_tag and variant.size_evidence is SizeEvidence.DERIVED
              else variant.model_id)
    caveat = None
    if variant.size_evidence is SizeEvidence.DERIVED:
        caveat = (f"{variant.quant} was derived, not measured: verify the "
                  f"'{tagged}' tag exists before pulling (peyk --online checks it).")
```

Use `tagged` in the `ollama run` line, add `"caveat": caveat` to the returned
dict, and have `report.render_snippet` print the caveat in `theme.YELLOW` when
it is not `None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/snippet.py src/peyk/report.py tests/test_plan_resolve_snippet.py
git commit -m "Flag snippets whose quantization tag is derived, not verified"
```

---

# Phase 2 — Language levels

### Task 9: `LanguageSupport` with legacy compatibility

**Files:**
- Create: `src/peyk/languages.py`
- Modify: `src/peyk/models.py:92`
- Test: `tests/test_languages.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Level` enum (`NATIVE`/`GOOD`/`PARTIAL`), `LangEvidence` enum (`BENCHMARK`/`STATED`/`INFERRED`/`UNKNOWN`), `LanguageSupport` model with `.supports(code) -> Level | None`, `ModelVariant.languages: LanguageSupport`, `ModelVariant.lang_evidence`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_languages.py
from peyk.languages import LangEvidence, LanguageSupport, Level
from peyk.models import ModelVariant


def test_explicit_levels_round_trip():
    ls = LanguageSupport(native=["en"], good=["tr", "de"], partial=["fa"])
    assert ls.supports("en") is Level.NATIVE
    assert ls.supports("tr") is Level.GOOD
    assert ls.supports("fa") is Level.PARTIAL
    assert ls.supports("sw") is None


def test_lookup_is_case_insensitive():
    assert LanguageSupport(good=["tr"]).supports("TR") is Level.GOOD


def test_legacy_flat_list_becomes_good_and_stays_closed():
    v = ModelVariant(model_id="x:7b", family="X", params_b=7, file_size_gb=4.0,
                     languages=["en", "de"])
    assert v.languages.supports("en") is Level.GOOD
    assert v.languages.supports("tr") is None      # an explicit list is a closed list


def test_legacy_multi_marker_downgrades_unlisted_languages_to_partial():
    v = ModelVariant(model_id="x:7b", family="X", params_b=7, file_size_gb=4.0,
                     languages=["multi"])
    assert v.languages.supports("tr") is Level.PARTIAL
    assert v.languages.catchall is Level.PARTIAL


def test_multi_plus_named_languages_keeps_the_named_ones_higher():
    v = ModelVariant(model_id="x:7b", family="X", params_b=7, file_size_gb=4.0,
                     languages=["multi", "en"])
    assert v.languages.supports("en") is Level.GOOD
    assert v.languages.supports("tr") is Level.PARTIAL


def test_lang_evidence_defaults_to_unknown():
    v = ModelVariant(model_id="x:7b", family="X", params_b=7, file_size_gb=4.0)
    assert v.lang_evidence is LangEvidence.UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_languages.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'peyk.languages'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/peyk/languages.py
"""Levelled language support with an evidence discount.

peyk ranks on language support, so "does it speak Turkish" must be more than a
boolean. Two axes carry that:

  level     how well the model handles the language (native > good > partial)
  evidence  how much we trust the claim (benchmark > stated > inferred > unknown)

The evidence axis mirrors `benchmarks.QualityEvidence` on purpose: peyk should
tell one consistent story about what it measured versus what it was told.

The legacy catalog marker "multi" is the reason `catchall` exists. A bare
"multi" is an unaudited claim of broad multilingualism; treating it as full
support let any such model score a perfect 100 for any language. It now resolves
unlisted languages to `partial` — not zero, because the claim is probably
directionally true, and not full, because nobody checked.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, model_validator

LEGACY_MULTI = "multi"


class Level(str, Enum):
    NATIVE = "native"
    GOOD = "good"
    PARTIAL = "partial"


class LangEvidence(str, Enum):
    BENCHMARK = "benchmark"
    STATED = "stated"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


LEVEL_WEIGHT: dict[Level, float] = {
    Level.NATIVE: 100.0,
    Level.GOOD: 80.0,
    Level.PARTIAL: 45.0,
}

EVIDENCE_FACTOR: dict[LangEvidence, float] = {
    LangEvidence.BENCHMARK: 1.00,
    LangEvidence.STATED: 0.95,
    LangEvidence.INFERRED: 0.80,
    LangEvidence.UNKNOWN: 0.60,
}


class LanguageSupport(BaseModel):
    native: list[str] = []
    good: list[str] = []
    partial: list[str] = []
    catchall: Level | None = None   # level for any code not listed above

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_list(cls, value):
        """Accept the pre-levels schema: a flat list of language codes."""
        if not isinstance(value, list):
            return value
        codes = [str(c).strip().lower() for c in value]
        named = [c for c in codes if c != LEGACY_MULTI]
        out: dict = {"good": named}
        if LEGACY_MULTI in codes:
            out["catchall"] = Level.PARTIAL
        return out

    def supports(self, code: str) -> Level | None:
        c = code.strip().lower()
        for level, bucket in ((Level.NATIVE, self.native),
                              (Level.GOOD, self.good),
                              (Level.PARTIAL, self.partial)):
            if c in (x.lower() for x in bucket):
                return level
        return self.catchall

    def all_codes(self) -> list[str]:
        return sorted({*self.native, *self.good, *self.partial})
```

In `src/peyk/models.py`, import `LangEvidence, LanguageSupport` from
`.languages` and replace the `languages` field:

```python
    languages: LanguageSupport = Field(default_factory=lambda: LanguageSupport(good=["en"]))
    lang_evidence: LangEvidence = LangEvidence.UNKNOWN
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_languages.py -v && ruff check . && mypy src/peyk`
Expected: new tests PASS. `pytest -q` will still fail in `test_scoring.py` — `language_score` reads `variant.languages` as a list. Task 10 fixes that; do not patch it here.

- [ ] **Step 5: Commit**

```bash
git add src/peyk/languages.py src/peyk/models.py tests/test_languages.py
git commit -m "Add levelled LanguageSupport that still accepts the legacy list schema"
```

---

### Task 10: Levelled language scoring

**Files:**
- Modify: `src/peyk/languages.py`, `src/peyk/scoring.py:51-58`
- Test: `tests/test_languages.py`

**Interfaces:**
- Consumes: `Level`, `LangEvidence`, `LEVEL_WEIGHT`, `EVIDENCE_FACTOR`, `LanguageSupport.supports`.
- Produces: `languages.score(support, evidence, wanted) -> float`; `scoring.language_score(variant, wanted)` delegating to it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_languages.py  (append)
import pytest

from peyk import languages
from peyk.scoring import language_score


def test_no_request_means_no_penalty():
    assert languages.score(LanguageSupport(good=["en"]), LangEvidence.STATED, []) == 100.0


def test_level_and_evidence_both_apply():
    ls = LanguageSupport(native=["en"], good=["tr"])
    # mean(100, 80) = 90, times the "stated" factor 0.95
    assert languages.score(ls, LangEvidence.STATED, ["en", "tr"]) == pytest.approx(85.5)


def test_benchmark_evidence_is_not_discounted():
    ls = LanguageSupport(native=["en"])
    assert languages.score(ls, LangEvidence.BENCHMARK, ["en"]) == pytest.approx(100.0)


def test_unsupported_language_scores_zero():
    ls = LanguageSupport(native=["en"])
    assert languages.score(ls, LangEvidence.BENCHMARK, ["tr"]) == 0.0


def test_unaudited_multi_no_longer_scores_full_marks():
    v = ModelVariant(model_id="x:7b", family="X", params_b=7, file_size_gb=4.0,
                     languages=["multi"])
    audited = ModelVariant(model_id="y:7b", family="Y", params_b=7, file_size_gb=4.0,
                           languages={"native": ["tr"]},
                           lang_evidence=LangEvidence.BENCHMARK)
    assert language_score(v, ["tr"]) < language_score(audited, ["tr"])
    assert language_score(v, ["tr"]) > 0        # a claim, not a refutation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_languages.py -k score -v`
Expected: FAIL — `AttributeError: module 'peyk.languages' has no attribute 'score'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/peyk/languages.py`:

```python
def score(support: LanguageSupport, evidence: LangEvidence, wanted: list[str]) -> float:
    """0-100 language fit: mean level weight over the requested languages,
    discounted by how well-evidenced the support claim is."""
    if not wanted:
        return 100.0
    total = 0.0
    for code in wanted:
        level = support.supports(code)
        total += LEVEL_WEIGHT[level] if level is not None else 0.0
    mean = total / len(wanted)
    return round(mean * EVIDENCE_FACTOR[evidence], 1)
```

Replace `language_score` in `src/peyk/scoring.py` (and add `from . import languages`):

```python
def language_score(variant: ModelVariant, wanted: list[str]) -> float:
    return _clamp(languages.score(variant.languages, variant.lang_evidence, wanted))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: PASS. Assertions in `tests/test_scoring.py` that expected a bare `["multi"]` variant to score 100 now expect a discounted value — **update them to the new intent**; that downgrade is the point of the change.

- [ ] **Step 5: Commit**

```bash
git add src/peyk/languages.py src/peyk/scoring.py tests/
git commit -m "Score language support by level and evidence instead of membership

A bare \"multi\" claim no longer scores a perfect 100 for every language."
```

---

### Task 11: Migrate the catalog's language data

**Files:**
- Modify: `src/peyk/sources/data/catalog.json`
- Create: `docs/languages.md`
- Test: `tests/test_sources.py`

**Interfaces:**
- Consumes: `LanguageSupport`, `LangEvidence`.
- Produces: a catalog where no variant relies on the `"multi"` fallback.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources.py  (append)
from peyk.languages import LangEvidence
from peyk.sources import build_catalog


def test_curated_catalog_has_no_unaudited_multi_claims():
    for cand in build_catalog(offline=True):
        for v in cand.variants:
            if v.source != "curated":
                continue
            assert v.languages.catchall is None, f"{v.model_id} still relies on 'multi'"
            assert v.lang_evidence is not LangEvidence.UNKNOWN, f"{v.model_id} lacks evidence"


def test_turkish_capable_models_are_findable():
    hits = [v for c in build_catalog(offline=True) for v in c.variants
            if v.languages.supports("tr") is not None]
    assert len(hits) >= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py -k multi -v`
Expected: FAIL — the 25 variants carrying `["multi"]` all have `catchall == Level.PARTIAL`.

- [ ] **Step 3: Write minimal implementation**

Rewrite each `languages` value in `src/peyk/sources/data/catalog.json` from a
flat list into the levelled object, and add `lang_evidence` per variant. Source
every level from the model's official card. Example, for the Qwen2.5 family:

```json
      "languages": {
        "native": ["en", "zh"],
        "good": ["tr", "fr", "de", "es", "it", "pt", "ru", "ja", "ko", "ar", "vi", "th", "id"],
        "partial": ["fa", "he", "hi", "bn", "ur"]
      },
      "lang_evidence": "stated",
```

and for the Llama 3.1/3.2/3.3 families (the card names eight languages and does
not claim Turkish):

```json
      "languages": {
        "native": ["en"],
        "good": ["de", "fr", "es", "it", "pt", "hi", "th"],
        "partial": ["tr", "nl", "pl", "ru"]
      },
      "lang_evidence": "stated",
```

Use `"lang_evidence": "benchmark"` only where a published multilingual benchmark
(Global-MMLU, Belebele, MMMLU) covers the family, and `"inferred"` where no card
claim exists but tokenizer/related-language evidence supports a `partial` entry.

Write `docs/languages.md` recording, per family: the level assignment, the model
card URL it came from, and the evidence tag chosen. This file is what makes the
curation reviewable — it is not optional.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/sources/data/catalog.json docs/languages.md tests/test_sources.py
git commit -m "Replace the catalog's 'multi' placeholder with levelled language data"
```

---

### Task 12: `Lang` column and `peyk languages`

**Files:**
- Modify: `src/peyk/report.py:94-133`, `src/peyk/cli.py`
- Test: `tests/test_engine_and_cli.py`

**Interfaces:**
- Consumes: `LanguageSupport.supports`, `Level`.
- Produces: `report.lang_cell(variant, wanted) -> Text`; `report.render_language_index(rows, console)`; `report.language_index_to_json(rows)`; CLI `languages` subcommand.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_and_cli.py  (append)
def test_lang_cell_marks_each_requested_language():
    from peyk.models import ModelVariant
    from peyk.report import lang_cell
    v = ModelVariant(model_id="x:7b", family="X", params_b=7, file_size_gb=4.0,
                     languages={"native": ["en"], "good": ["tr"]})
    assert lang_cell(v, ["en", "tr", "sw"]).plain == "en● tr◐ sw·"


def test_languages_command_lists_the_catalog(capsys):
    import json
    from peyk.cli import main
    assert main(["languages", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert any(row["language"] == "tr" for row in data["languages"])


def test_languages_command_filters_to_one_language(capsys):
    import json
    from peyk.cli import main
    assert main(["languages", "tr", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["language"] == "tr"
    assert data["models"] and all("level" in m for m in data["models"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine_and_cli.py -k lang -v`
Expected: FAIL — `ImportError: cannot import name 'lang_cell' from 'peyk.report'`

- [ ] **Step 3: Write minimal implementation**

In `src/peyk/report.py`:

```python
_LEVEL_GLYPH = {Level.NATIVE: "●", Level.GOOD: "◐", Level.PARTIAL: "○"}
_LEVEL_STYLE = {Level.NATIVE: "green", Level.GOOD: "cyan", Level.PARTIAL: "yellow"}


def lang_cell(variant, wanted: list[str], plain: bool = False) -> Text:
    """One glyph per requested language: ● native, ◐ good, ○ partial, · absent."""
    out = Text()
    for i, code in enumerate(wanted):
        if i:
            out.append(" ")
        level = variant.languages.supports(code)
        if plain:
            out.append(f"{code}:{level.value if level else 'no'}")
        else:
            out.append(code, style=theme.MUTED)
            out.append(_LEVEL_GLYPH.get(level, "·"),
                       style=_LEVEL_STYLE.get(level, theme.MUTED))
    return out
```

Add a `Lang` column to `_tier_table` between `Quant` and `Memory`, populated with
`lang_cell(v, rec.languages)`. Store the requested languages on `Recommendation`
(`self.languages = languages` in `engine.Recommendation.__init__`, passed from
`recommend`) so the renderer knows what to mark. Extend `_tier_legend()` with
`"● native  ◐ good  ○ partial  · unsupported"`.

Add the index renderer and its JSON twin (`render_language_index`,
`language_index_to_json`), then the CLI subparser:

```python
    lg = sub.add_parser("languages", help="Which languages the catalog covers")
    lg.add_argument("language", nargs="?", help="Filter to one code, e.g. tr")
    lg.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    _add_common_flags(lg)
```

```python
def _cmd_languages(args, console: Console, status: Console) -> int:
    from .languages import Level
    variants = [v for c in build_catalog(offline=True) for v in c.variants]
    if args.language:
        code = args.language.strip().lower()
        models = []
        for v in variants:
            level = v.languages.supports(code)
            if level is None:
                continue
            models.append({"model_id": v.model_id, "family": v.family,
                           "params_b": v.params_b, "level": level.value,
                           "evidence": v.lang_evidence.value,
                           "quality": v.quality_score})
        models.sort(key=lambda m: (-_LEVEL_RANK[m["level"]], -m["quality"]))
        payload = {"language": code, "models": models}
        if args.json:
            import json
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            report.render_language_models(payload, console=console)
        return 0
    counts: dict[str, dict[str, int]] = {}
    for v in variants:
        for code in v.languages.all_codes():
            level = v.languages.supports(code)
            if level is None:
                continue
            counts.setdefault(code, {l.value: 0 for l in Level})[level.value] += 1
    rows = [{"language": code, **levels} for code, levels in sorted(counts.items())]
    if args.json:
        import json
        print(json.dumps({"languages": rows}, indent=2, ensure_ascii=False))
    else:
        report.render_language_index(rows, console=console)
    return 0
```

with `_LEVEL_RANK = {"native": 3, "good": 2, "partial": 1}` module-level in
`cli.py`. Add `"languages"` to `SUBCOMMANDS` and `_DISPATCH`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/report.py src/peyk/engine.py src/peyk/cli.py tests/test_engine_and_cli.py
git commit -m "Show per-language support in the report and add peyk languages"
```

---

# Phase 3 — Roles: embedding and reranker

### Task 13: Role-aware memory estimation

**Files:**
- Modify: `src/peyk/estimator.py:11-24`, `src/peyk/estimator.py:27-42`, `src/peyk/estimator.py:97-111`
- Test: `tests/test_roles.py`

**Interfaces:**
- Consumes: `Role`, `FitResult.index_gb`.
- Produces: `ROLE_OVERHEAD_GB: dict[Role, float]`; `index_gb(variant, chunks) -> float`; `estimate_fit(variant, hw, context, kv_bits=16, chunks=0)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_roles.py
import pytest

from peyk.estimator import estimate_fit, index_gb, kv_cache_gb
from peyk.models import ModelVariant, Role


def _embedder(**kw) -> ModelVariant:
    base = dict(model_id="bge-m3", family="BGE-M3", params_b=0.568,
                quant="Q8_0", file_size_gb=1.2, context_max=8192,
                role=Role.EMBEDDING, embed_dim=1024)
    base.update(kw)
    return ModelVariant(**base)


def test_encoders_have_no_kv_cache():
    # Embedding and reranker models are encoders: no autoregressive cache.
    assert kv_cache_gb(8192, _embedder()) == 0.0
    assert kv_cache_gb(8192, _embedder(role=Role.RERANKER, embed_dim=None)) == 0.0


def test_encoders_use_a_lighter_runtime_overhead(laptop_cpu):
    fit = estimate_fit(_embedder(), laptop_cpu, 8192)
    assert fit.overhead_gb == 0.3
    assert fit.mem_need_gb == pytest.approx(1.5, abs=0.01)   # 1.2 weights + 0.3


def test_llm_overhead_is_unchanged(small_variant, laptop_cpu):
    assert estimate_fit(small_variant, laptop_cpu, 8192).overhead_gb == 0.9


def test_index_cost_scales_with_dimension_and_chunks():
    # fp32 vectors: 1M chunks x 1024 dims x 4 bytes = 4.096 GB
    assert index_gb(_embedder(), 1_000_000) == pytest.approx(4.096, rel=1e-3)
    assert index_gb(_embedder(embed_dim=384), 1_000_000) == pytest.approx(1.536, rel=1e-3)
    assert index_gb(_embedder(), 0) == 0.0


def test_index_cost_is_zero_for_non_embedding_roles(small_variant):
    assert index_gb(small_variant, 1_000_000) == 0.0


def test_fit_reports_the_index_alongside_the_weights(laptop_cpu):
    fit = estimate_fit(_embedder(), laptop_cpu, 8192, chunks=1_000_000)
    assert fit.index_gb == pytest.approx(4.096, rel=1e-3)
    assert fit.mem_need_gb == pytest.approx(1.5 + 4.096, abs=0.02)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_roles.py -v`
Expected: FAIL — `ImportError: cannot import name 'index_gb' from 'peyk.estimator'`

- [ ] **Step 3: Write minimal implementation**

In `src/peyk/estimator.py`, add `Role` to the `.models` import and:

```python
# Serving overhead outside weights + KV. Encoder stacks (embedding, reranking)
# run far lighter runtimes than a generation server.
ROLE_OVERHEAD_GB = {Role.LLM: 0.9, Role.EMBEDDING: 0.3, Role.RERANKER: 0.3}
BYTES_PER_VECTOR_ELEMENT = 4  # fp32 index
```

Guard the KV cache and add the index cost:

```python
def kv_cache_gb(context: int, variant: ModelVariant, kv_bits: int = 16) -> float:
    """... (docstring as in Task 5, plus:)

    Encoders (embedding, reranker) have no autoregressive cache, so this is 0.
    """
    if variant.role is not Role.LLM:
        return 0.0
    ...


def index_gb(variant: ModelVariant, chunks: int) -> float:
    """Vector-index cost of embedding `chunks` documents with this model.

    An embedding model's real cost is usually its index, not its weights: at
    dim 1024, a million chunks is 4.1 GB — larger than any embedding model in
    the catalog.
    """
    if variant.role is not Role.EMBEDDING or not variant.embed_dim or chunks <= 0:
        return 0.0
    return chunks * variant.embed_dim * BYTES_PER_VECTOR_ELEMENT / 1e9
```

and rewrite `estimate_fit`:

```python
def estimate_fit(
    variant: ModelVariant, hw: HardwareProfile, context: int,
    kv_bits: int = 16, chunks: int = 0,
) -> FitResult:
    ctx = min(context, variant.context_max)
    kv = kv_cache_gb(ctx, variant, kv_bits)
    overhead = ROLE_OVERHEAD_GB.get(variant.role, RUNTIME_OVERHEAD_GB)
    idx = index_gb(variant, chunks)
    mem_need = variant.file_size_gb + kv + overhead + idx
    tier = classify(mem_need, hw.memory_pool_gb)
    tps = estimate_tokens_per_sec(variant, hw, _effective_bandwidth(mem_need, hw))
    return FitResult(
        variant=variant,
        mem_need_gb=round(mem_need, 2),
        tier=tier,
        est_tokens_per_sec=round(tps, 1),
        weights_gb=round(variant.file_size_gb, 2),
        kv_cache_gb=round(kv, 2),
        overhead_gb=overhead,
        index_gb=round(idx, 3),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/estimator.py tests/test_roles.py
git commit -m "Size encoders correctly: no KV cache, lighter overhead, index cost"
```

---

### Task 14: Role-specific scoring criteria

**Files:**
- Modify: `src/peyk/scoring.py:20-35`, `src/peyk/scoring.py:81-128`, `src/peyk/engine.py:11`
- Test: `tests/test_roles.py`

**Interfaces:**
- Consumes: `Role`, `index_gb`.
- Produces: `scoring.CRITERIA_BY_ROLE: dict[Role, list[str]]`; `scoring.ROLE_WEIGHTS: dict[Role, dict[str, float]]`; `scoring.index_cost_score(variant) -> float`; `engine.criteria_for(role) -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_roles.py  (append)
from peyk.engine import criteria_for
from peyk.scoring import index_cost_score, score_variant, weights_for


def test_criteria_differ_by_role():
    assert criteria_for(Role.LLM) == ["speed", "quality", "language", "context", "license"]
    assert "index_cost" in criteria_for(Role.EMBEDDING)
    assert "speed" not in criteria_for(Role.EMBEDDING)
    assert "speed" in criteria_for(Role.RERANKER)
    assert "index_cost" not in criteria_for(Role.RERANKER)


def test_smaller_vectors_score_better_on_index_cost():
    assert index_cost_score(_embedder(embed_dim=384)) > index_cost_score(_embedder(embed_dim=1024))
    assert index_cost_score(_embedder(embed_dim=4096)) >= 0.0


def test_index_cost_is_neutral_for_non_embedding_roles(small_variant):
    assert index_cost_score(small_variant) == 100.0


def test_embedding_scores_carry_only_its_own_criteria(laptop_cpu):
    scored = score_variant(_embedder(), laptop_cpu, 8192, ["tr"],
                           weights_for(None, Role.EMBEDDING))
    assert set(scored.scores) == set(criteria_for(Role.EMBEDDING))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_roles.py -k criteria -v`
Expected: FAIL — `ImportError: cannot import name 'criteria_for' from 'peyk.engine'`

- [ ] **Step 3: Write minimal implementation**

In `src/peyk/scoring.py`:

```python
# An embedding model's index dominates its cost, so dimension is a first-class
# criterion; encoders have no generation throughput worth ranking.
CRITERIA_BY_ROLE: dict[Role, list[str]] = {
    Role.LLM: ["speed", "quality", "language", "context", "license"],
    Role.EMBEDDING: ["quality", "language", "context", "license", "index_cost"],
    Role.RERANKER: ["quality", "language", "context", "license", "speed"],
}

ROLE_WEIGHTS: dict[Role, dict[str, float]] = {
    Role.EMBEDDING: {"quality": 0.40, "language": 0.25, "context": 0.15,
                     "license": 0.10, "index_cost": 0.10},
    Role.RERANKER: {"quality": 0.45, "language": 0.25, "context": 0.10,
                    "license": 0.10, "speed": 0.10},
}

# Vector width that scores 0; anything narrower scores proportionally higher.
INDEX_DIM_REFERENCE = 2048


def index_cost_score(variant: ModelVariant) -> float:
    """Smaller vectors cost less to store and search, so they score higher."""
    if variant.role is not Role.EMBEDDING or not variant.embed_dim:
        return 100.0
    return _clamp((1.0 - variant.embed_dim / INDEX_DIM_REFERENCE) * 100.0)
```

`weights_for` gains a role argument:

```python
def weights_for(use_case: str | None, role: Role = Role.LLM) -> dict[str, float]:
    if role is not Role.LLM:
        return ROLE_WEIGHTS[role]
    if use_case and use_case in USE_CASE_WEIGHTS:
        return USE_CASE_WEIGHTS[use_case]
    return DEFAULT_WEIGHTS
```

`score_variant` builds only the criteria its role uses:

```python
    all_scores = {
        "speed": round(speed_score(fit), 1),
        "quality": round(_clamp(evidence.effective * quant.retention(variant.quant)), 1),
        "language": round(language_score(variant, languages), 1),
        "context": round(context_score(variant), 1),
        "license": round(license_score(variant), 1),
        "index_cost": round(index_cost_score(variant), 1),
    }
    scores = {k: all_scores[k] for k in CRITERIA_BY_ROLE[variant.role]}
```

In `src/peyk/engine.py`, replace the module-level `CRITERIA` list with:

```python
def criteria_for(role: Role = Role.LLM) -> list[str]:
    return list(scoring.CRITERIA_BY_ROLE[role])


CRITERIA = criteria_for(Role.LLM)  # kept: report.py iterates it for LLM tables
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/scoring.py src/peyk/engine.py tests/test_roles.py
git commit -m "Score embedding and reranker models on criteria that apply to them"
```

---

### Task 15: Catalog entries for embedding and reranker models

**Files:**
- Modify: `src/peyk/sources/data/catalog.json`, `docs/languages.md`
- Test: `tests/test_roles.py`

**Interfaces:**
- Consumes: `Role`, `LanguageSupport`, `embed_dim`.
- Produces: 11 embedding + 5 reranker catalog variants.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_roles.py  (append)
from peyk.languages import LangEvidence
from peyk.sources import build_catalog


def _by_role(role):
    return [v for c in build_catalog(offline=True) for v in c.variants if v.role is role]


def test_catalog_carries_embedding_models():
    embedders = _by_role(Role.EMBEDDING)
    assert len(embedders) >= 10
    assert all(v.embed_dim and v.embed_dim > 0 for v in embedders)
    assert any("bge-m3" in v.model_id for v in embedders)


def test_catalog_carries_rerankers():
    assert len(_by_role(Role.RERANKER)) >= 5


def test_turkish_capable_embedders_exist():
    hits = [v for v in _by_role(Role.EMBEDDING) if v.languages.supports("tr")]
    assert len(hits) >= 4


def test_new_entries_declare_language_evidence():
    for v in _by_role(Role.EMBEDDING) + _by_role(Role.RERANKER):
        assert v.lang_evidence is not LangEvidence.UNKNOWN


def test_non_commercial_licenses_are_recorded_honestly():
    jina = [v for v in _by_role(Role.EMBEDDING) if "jina" in v.model_id]
    assert jina and all("nc" in v.license.lower() for v in jina)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_roles.py -k catalog -v`
Expected: FAIL — `assert 0 >= 10`

- [ ] **Step 3: Write minimal implementation**

Append 16 variants to `src/peyk/sources/data/catalog.json`. Each carries
`role`, `embed_dim` (embedding only), levelled `languages`, `lang_evidence`, and
an honest `license`. One embedding and one reranker entry shown in full; follow
the same shape for the rest.

```json
    {
      "model_id": "bge-m3",
      "family": "BGE-M3",
      "params_b": 0.568,
      "quant": "Q8_0",
      "file_size_gb": 1.2,
      "context_max": 8192,
      "role": "embedding",
      "embed_dim": 1024,
      "languages": {
        "native": ["en", "zh"],
        "good": ["tr", "fr", "de", "es", "it", "pt", "ru", "ja", "ko", "ar", "hi", "th", "vi", "id", "nl", "pl"],
        "partial": ["fa", "he", "bn", "ur", "sw"]
      },
      "lang_evidence": "benchmark",
      "license": "mit",
      "quality_score": 78
    },
    {
      "model_id": "bge-reranker-v2-m3",
      "family": "BGE-Reranker-v2-M3",
      "params_b": 0.568,
      "quant": "Q8_0",
      "file_size_gb": 1.2,
      "context_max": 8192,
      "role": "reranker",
      "languages": {
        "native": ["en", "zh"],
        "good": ["tr", "fr", "de", "es", "it", "pt", "ru", "ja", "ko", "ar", "hi", "th", "vi", "id"],
        "partial": ["fa", "he", "bn", "ur"]
      },
      "lang_evidence": "benchmark",
      "license": "apache-2.0",
      "quality_score": 80
    },
```

Full roster to add — **embedding (11):** bge-m3, multilingual-e5-large,
multilingual-e5-base, Qwen3-Embedding-0.6B, Qwen3-Embedding-4B,
Qwen3-Embedding-8B, nomic-embed-text-v1.5, gte-multilingual-base,
embeddinggemma-300m, jina-embeddings-v3 (`"license": "cc-by-nc-4.0"`),
snowflake-arctic-embed-l-v2.0, all-MiniLM-L6-v2. **Reranker (5):**
bge-reranker-v2-m3, Qwen3-Reranker-0.6B, Qwen3-Reranker-4B,
jina-reranker-v2-base-multilingual (`"license": "cc-by-nc-4.0"`),
mxbai-rerank-base-v2.

`quality_score` for these comes from MTEB/MIRACL retrieval scores normalised to
0-100; record the source in `docs/languages.md` alongside the language rows so
one file explains every hand-curated number for these entries.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/sources/data/catalog.json docs/languages.md tests/test_roles.py
git commit -m "Add embedding and reranker models to the curated catalog"
```

---

### Task 16: `--role` filter and the `--use-case embedding` deprecation

**Files:**
- Modify: `src/peyk/engine.py:32-56`, `src/peyk/cli.py`
- Test: `tests/test_roles.py`

**Interfaces:**
- Consumes: `criteria_for`, `weights_for(use_case, role)`.
- Produces: `recommend(..., role: Role = Role.LLM)`; `--role` on `recommend`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_roles.py  (append)
def test_recommend_defaults_to_llms_only(laptop_cpu):
    from peyk.engine import recommend
    rec = recommend(hw=laptop_cpu, candidates=build_catalog(offline=True), context=8192)
    assert all(s.variant.role is Role.LLM for s in rec.scored)


def test_recommend_can_target_embedders(laptop_cpu):
    from peyk.engine import recommend
    rec = recommend(hw=laptop_cpu, candidates=build_catalog(offline=True),
                    context=8192, role=Role.EMBEDDING)
    assert rec.scored and all(s.variant.role is Role.EMBEDDING for s in rec.scored)


def test_use_case_embedding_is_a_deprecated_alias_for_role(capsys):
    from peyk.cli import main
    assert main(["recommend", "--use-case", "embedding", "--json"]) == 0
    err = capsys.readouterr().err
    assert "deprecated" in err.lower() and "--role embedding" in err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_roles.py -k role -v`
Expected: FAIL — `TypeError: recommend() got an unexpected keyword argument 'role'`

- [ ] **Step 3: Write minimal implementation**

In `src/peyk/engine.py`, add `role: Role = Role.LLM` to `recommend`, filter
candidates by it, and pass the role to `weights_for` and
`best_runnable_variant`:

```python
    weights = weights_for(use_case, role)
    ...
    for cand in candidates:
        variants = [v for v in quant.expand(cand, hw, only_quant=only_quant,
                                            only_format=only_format)
                    if v.role is role]
```

In `src/peyk/cli.py`, add the flag and the alias:

```python
    rec.add_argument("--role", choices=[r.value for r in Role], default="llm",
                     help="Model role to rank: llm (default), embedding, reranker")
```

and in `_cmd_recommend`, before calling `recommend`:

```python
    role = Role(args.role)
    if args.use_case == "embedding":
        status.print("[yellow]--use-case embedding is deprecated; "
                     "use --role embedding instead.[/yellow]")
        role, args.use_case = Role.EMBEDDING, None
```

Remove `"embedding"` from `USE_CASES` in `cli.py` but keep accepting it: declare
`--use-case` with `choices=USE_CASES + ["embedding"]` and leave `"embedding"`
out of the help text.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/engine.py src/peyk/cli.py tests/test_roles.py
git commit -m "Add --role and deprecate --use-case embedding in favour of it"
```

---

### Task 17: `peyk rag` — a co-resident stack

**Files:**
- Create: `src/peyk/rag.py`
- Modify: `src/peyk/cli.py`, `src/peyk/report.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `recommend`, `estimate_fit`, `index_gb`, `Role`.
- Produces: `rag.RagStack` dataclass (`embedding`, `reranker`, `llm`, `total_gb`, `index_gb`, `fits`); `rag.select_stack(hw, candidates, context, languages, chunks, with_reranker) -> RagStack | None`; `report.render_rag(stack, console)`; `report.rag_to_json(stack)`; CLI `rag` subcommand.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag.py
import pytest

from peyk import rag
from peyk.models import Accelerator, HardwareProfile, ModelCandidate, ModelVariant, Role
from peyk.sources import build_catalog


def _tiny_box(gb: float) -> HardwareProfile:
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores_physical=4,
                           cpu_cores_logical=8, ram_total_gb=gb,
                           ram_available_gb=gb * 0.6,
                           accelerator=Accelerator.NONE, mem_bandwidth_gbs=50.0)


def test_stack_picks_one_model_per_role(laptop_cpu):
    stack = rag.select_stack(laptop_cpu, build_catalog(offline=True),
                             context=8192, languages=["tr"], chunks=0)
    assert stack is not None
    assert stack.embedding.variant.role is Role.EMBEDDING
    assert stack.reranker.variant.role is Role.RERANKER
    assert stack.llm.variant.role is Role.LLM


def test_stack_total_is_the_sum_of_its_parts(laptop_cpu):
    stack = rag.select_stack(laptop_cpu, build_catalog(offline=True),
                             context=8192, languages=["tr"], chunks=0)
    parts = stack.embedding.fit.mem_need_gb + stack.reranker.fit.mem_need_gb \
        + stack.llm.fit.mem_need_gb
    assert stack.total_gb == pytest.approx(parts, abs=0.01)


def test_stack_respects_the_shared_memory_pool():
    # 6 GB box: the three models must co-reside, not each fit alone.
    hw = _tiny_box(6.0)
    stack = rag.select_stack(hw, build_catalog(offline=True),
                             context=4096, languages=["en"], chunks=0)
    assert stack is not None
    assert stack.fits
    assert stack.total_gb <= hw.memory_pool_gb


def test_index_cost_counts_against_the_pool():
    hw = _tiny_box(8.0)
    lean = rag.select_stack(hw, build_catalog(offline=True), context=4096,
                            languages=["en"], chunks=0)
    heavy = rag.select_stack(hw, build_catalog(offline=True), context=4096,
                             languages=["en"], chunks=2_000_000)
    assert heavy.index_gb > 0
    assert heavy.total_gb > lean.total_gb


def test_reranker_can_be_omitted(laptop_cpu):
    stack = rag.select_stack(laptop_cpu, build_catalog(offline=True), context=8192,
                             languages=["en"], chunks=0, with_reranker=False)
    assert stack.reranker is None


def test_impossible_budget_reports_no_fit():
    stack = rag.select_stack(_tiny_box(1.0), build_catalog(offline=True),
                             context=4096, languages=["en"], chunks=0)
    assert stack is None or not stack.fits
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'peyk.rag'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/peyk/rag.py
"""Pick an embedding + reranker + LLM stack that fits in memory *together*.

Ranking each role independently and stapling the winners together is the wrong
answer: a RAG deployment runs all three at once, and their memory adds up — as
does the vector index, which at dim 1024 and a million chunks outweighs every
embedding model in the catalog.

The search space is tiny (tens of models per role), so this does an exhaustive
search over the top candidates of each role. Deterministic, no solver.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import recommend
from .models import HardwareProfile, ModelCandidate, Role, ScoredModel

# How many per-role candidates enter the combination search. The roles hold tens
# of models, so this is generous while keeping the search trivially small.
TOP_PER_ROLE = 8


@dataclass(frozen=True)
class RagStack:
    embedding: ScoredModel
    llm: ScoredModel
    reranker: ScoredModel | None
    total_gb: float
    index_gb: float
    pool_gb: float

    @property
    def fits(self) -> bool:
        return self.total_gb <= self.pool_gb

    @property
    def score(self) -> float:
        parts = [self.embedding.overall, self.llm.overall]
        if self.reranker is not None:
            parts.append(self.reranker.overall)
        return round(sum(parts) / len(parts), 1)


def _candidates_for(hw, catalog, context, languages, role, chunks=0):
    rec = recommend(hw=hw, candidates=catalog, context=context,
                    languages=languages, role=role, chunks=chunks)
    return rec.overall_top(TOP_PER_ROLE) or rec.scored[:TOP_PER_ROLE]


def select_stack(
    hw: HardwareProfile,
    catalog: list[ModelCandidate],
    context: int,
    languages: list[str],
    chunks: int = 0,
    with_reranker: bool = True,
) -> RagStack | None:
    """Highest-scoring triple whose combined memory fits the pool."""
    embedders = _candidates_for(hw, catalog, context, languages, Role.EMBEDDING, chunks)
    llms = _candidates_for(hw, catalog, context, languages, Role.LLM)
    rerankers = (_candidates_for(hw, catalog, context, languages, Role.RERANKER)
                 if with_reranker else [None])
    if not embedders or not llms or not rerankers:
        return None

    pool = hw.memory_pool_gb
    best: RagStack | None = None
    for emb in embedders:
        for llm in llms:
            for rr in rerankers:
                total = emb.fit.mem_need_gb + llm.fit.mem_need_gb
                if rr is not None:
                    total += rr.fit.mem_need_gb
                stack = RagStack(embedding=emb, llm=llm, reranker=rr,
                                 total_gb=round(total, 2),
                                 index_gb=emb.fit.index_gb, pool_gb=pool)
                if best is None:
                    best = stack
                    continue
                # Prefer a fitting stack; among equals, prefer the better score.
                if (stack.fits, stack.score) > (best.fits, best.score):
                    best = stack
    return best
```

`recommend` needs a `chunks: int = 0` parameter threaded into
`best_runnable_variant` → `score_variant` → `estimate_fit`, so the index cost is
inside `fit.mem_need_gb` for embedding candidates. Add it alongside `kv_bits`
with the same plumbing.

In `src/peyk/cli.py`, add `"rag"` to `SUBCOMMANDS`/`_DISPATCH` and:

```python
    rg = sub.add_parser("rag", help="Pick a co-resident embedding + reranker + LLM stack")
    rg.add_argument("--languages", default=str(cfg.get("languages", "en")),
                    help="Comma-separated language codes, e.g. tr,en")
    rg.add_argument("--context", type=int, default=int(cfg.get("context", 8192)),
                    help="Target context length in tokens")
    rg.add_argument("--chunks", type=int, default=0,
                    help="Documents to index — sizes the vector index")
    rg.add_argument("--no-reranker", action="store_true", help="Skip the reranker stage")
    rg.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    _add_hardware_flags(rg)
    _add_common_flags(rg)
```

```python
def _cmd_rag(args, console: Console, status: Console) -> int:
    from .rag import select_stack
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    hw = _resolve_hardware(args, status)
    stack = select_stack(hw, build_catalog(offline=True), context=args.context,
                         languages=languages, chunks=args.chunks,
                         with_reranker=not args.no_reranker)
    if stack is None:
        status.print("[red]No RAG stack could be assembled from the catalog.[/red]")
        return 1
    if args.json:
        print(report.rag_to_json(stack))
    else:
        report.render_rag(stack, console=console)
    return 0 if stack.fits else 1
```

`report.render_rag` prints one row per role (role, model, params, quant, memory,
and `dim`/`index` for the embedder), then the total against the pool and an
explicit fits/does-not-fit verdict.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/rag.py src/peyk/cli.py src/peyk/report.py src/peyk/engine.py \
        src/peyk/scoring.py tests/test_rag.py
git commit -m "Add peyk rag: an embedding + reranker + LLM stack that co-resides"
```

---

# Phase 4 — CLI simplification

### Task 18: Deprecated-alias infrastructure and the new flags

**Files:**
- Modify: `src/peyk/cli.py:39-132`, `src/peyk/cli.py:150-215`
- Test: `tests/test_cli_flags.py`

**Interfaces:**
- Consumes: the existing parser.
- Produces: `cli.DEPRECATED: dict[str, str]`; `cli.apply_deprecations(args, status) -> None`; new flags `--online`, `--source`, `--benchmarks`, `--refresh`, `--probe`, `--memory`, `--reserve`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_flags.py
import pytest

from peyk.cli import main


@pytest.mark.parametrize("old,new", [
    (["--cross-check"], ["--online"]),
    (["--discover"], ["--online", "discover"]),
    (["--no-cache"], ["--refresh"]),
    (["--deep"], ["--probe"]),
    (["--ram-budget", "8GB"], ["--memory", "8GB"]),
    (["--vram-headroom", "1GB"], ["--reserve", "1GB"]),
])
def test_deprecated_flags_still_work(old, new, capsys):
    assert main(["recommend", "--gpu", "RTX 4090", "--json", *old]) == 0
    old_out = capsys.readouterr()
    assert main(["recommend", "--gpu", "RTX 4090", "--json", *new]) == 0
    new_out = capsys.readouterr()
    assert old_out.out == new_out.out
    assert "deprecated" in old_out.err.lower()
    assert "deprecated" not in new_out.err.lower()


def test_online_defaults_to_cross_check_plus_benchmarks(capsys):
    assert main(["recommend", "--gpu", "RTX 4090", "--online", "--json"]) == 0


def test_probe_deep_implies_sudo(capsys):
    from peyk.cli import _build_parser
    args = _build_parser({}).parse_args(["recommend", "--probe", "deep"])
    assert args.deep is True and args.sudo is True


def test_memory_and_reserve_are_parsed_as_sizes():
    from peyk.cli import _build_parser
    args = _build_parser({}).parse_args(
        ["recommend", "--memory", "24GB", "--reserve", "1.5GB"])
    assert args.ram_budget == "24GB" and args.vram_headroom == "1.5GB"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_flags.py -v`
Expected: FAIL — argparse exits 2 with "unrecognized arguments: --online"

- [ ] **Step 3: Write minimal implementation**

New flags write into the **existing** destinations, so no downstream code
changes; old flags keep their own destinations and are hidden.

```python
DEPRECATED = {
    "cross_check": "--online",
    "discover": "--online discover",
    "live_benchmarks": "--online",
    "no_cache": "--refresh",
    "catalog_url": "--source",
    "benchmarks_url": "--benchmarks",
    "deep": "--probe",
    "sudo": "--probe deep",
    "offline": "(offline is the default; the flag is no longer needed)",
    "vram_headroom": "--reserve",
    "ram_budget": "--memory",
}
```

In `_add_hardware_flags`, keep `--deep`/`--sudo` but with `help=argparse.SUPPRESS`
and `dest` unchanged, and add:

```python
    p.add_argument("--probe", nargs="?", const="basic", choices=["basic", "deep"],
                   help="Run native probe scripts; 'deep' adds sudo (Linux dmidecode)")
```

In the `recommend` parser, suppress the seven old source flags and the two old
budget flags, then add:

```python
    src = p.add_argument_group("Sources")
    src.add_argument("--online", nargs="?", const="cross-check",
                     choices=["cross-check", "discover", "all"],
                     help="Use live sources: cross-check sizes (default), "
                          "'discover' adds HuggingFace discovery, 'all' both")
    src.add_argument("--source", metavar="URL", dest="catalog_url_new",
                     help="Fetch the catalog from a remote endpoint")
    src.add_argument("--benchmarks", metavar="URL", dest="benchmarks_url_new",
                     help="JSON endpoint for the live benchmark tier")
    src.add_argument("--refresh", action="store_true",
                     help="Bypass the ~/.cache/peyk TTL cache")
```

Normalise in one place, called at the top of `main` after parsing:

```python
def apply_deprecations(args, status: Console) -> None:
    """Fold new flags onto the original destinations and warn about old ones."""
    used = [flag for dest, flag in DEPRECATED.items() if getattr(args, dest, None)]
    if used:
        status.print(f"[yellow]Deprecated flag(s) in use; prefer: "
                     f"{', '.join(sorted(set(used)))}[/yellow]")

    online = getattr(args, "online", None)
    if online:
        args.cross_check = True
        args.live_benchmarks = True
        if online in ("discover", "all"):
            args.discover = True
    probe = getattr(args, "probe", None)
    if probe:
        args.deep = True
        if probe == "deep":
            args.sudo = True
    for new, old in (("catalog_url_new", "catalog_url"),
                     ("benchmarks_url_new", "benchmarks_url")):
        if getattr(args, new, None):
            setattr(args, old, getattr(args, new))
    if getattr(args, "refresh", False):
        args.no_cache = True
    if getattr(args, "memory", None):
        args.ram_budget = args.memory
    if getattr(args, "reserve", None):
        args.vram_headroom = args.reserve
```

Add `--memory` / `--reserve` to the `recommend` "Filters" group with
`dest="memory"` / `dest="reserve"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/cli.py tests/test_cli_flags.py
git commit -m "Collapse the source and probe flag clusters, keeping old names working"
```

---

### Task 19: Help groups and config keys

**Files:**
- Modify: `src/peyk/cli.py:55-132`, `src/peyk/config.py:19`
- Test: `tests/test_config.py`, `tests/test_cli_flags.py`

**Interfaces:**
- Consumes: the parser from Task 18.
- Produces: four argument groups on `recommend`; `CONFIG_KEYS` extended with `role`, `quant`, `format`, `kv_quant`, `chunks`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_flags.py  (append)
def test_help_is_grouped():
    from peyk.cli import _build_parser
    sub = _build_parser({})._subparsers._group_actions[0].choices["recommend"]
    titles = {g.title for g in sub._action_groups}
    assert {"Hardware", "Sources", "Filters", "Output"} <= titles


def test_deprecated_flags_are_hidden_from_help():
    from peyk.cli import _build_parser
    sub = _build_parser({})._subparsers._group_actions[0].choices["recommend"]
    text = sub.format_help()
    for hidden in ("--cross-check", "--discover", "--no-cache", "--ram-budget"):
        assert hidden not in text
    for shown in ("--online", "--refresh", "--memory", "--role", "--kv-quant"):
        assert shown in text
```

```python
# tests/test_config.py  (append)
def test_config_accepts_the_new_keys(tmp_path, monkeypatch):
    import json
    from peyk.config import load_config
    (tmp_path / ".peyk.json").write_text(json.dumps({
        "role": "embedding", "quant": "Q6_K", "format": "gguf",
        "kv_quant": "q8_0", "chunks": 500000, "nonsense": 1,
    }))
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg["role"] == "embedding" and cfg["kv_quant"] == "q8_0"
    assert "nonsense" not in cfg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_flags.py -k help tests/test_config.py -v`
Expected: FAIL — group titles are argparse's defaults; `role` is filtered out of the config.

- [ ] **Step 3: Write minimal implementation**

In `_build_parser`, create the groups once and register each `recommend`
argument on the right one:

```python
    hardware = rec.add_argument_group("Hardware")
    sources = rec.add_argument_group("Sources")
    filters = rec.add_argument_group("Filters")
    output = rec.add_argument_group("Output")
```

- **Hardware:** `--gpu`, `--gpu-only`, `--probe`, `--memory`, `--reserve`
- **Sources:** `--online`, `--source`, `--benchmarks`, `--refresh`
- **Filters:** `--use-case`, `--role`, `--languages`, `--context`, `--quant`, `--format`, `--kv-quant`, `--speed`, `--all`
- **Output:** `--top`, `--json`, `--markdown`, `--plain`

Refactor `_add_hardware_flags(p, group=None)` and `_add_common_flags(p, group=None)`
to add to the given group when one is passed, so the other subcommands are
unaffected.

In `src/peyk/config.py`:

```python
CONFIG_KEYS = {"languages", "use_case", "context", "top", "catalog_url",
               "role", "quant", "format", "kv_quant", "chunks"}
```

and use each as a default in `_build_parser` the way `languages` already is.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q && ruff check . && mypy src/peyk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peyk/cli.py src/peyk/config.py tests/test_cli_flags.py tests/test_config.py
git commit -m "Group recommend's flags into Hardware/Sources/Filters/Output"
```

---

### Task 20: Documentation and regenerated examples

**Files:**
- Modify: `README.md`, `docs/design.md`, `docs/roadmap.md`, `examples/README.md`, `examples/report.json`, `examples/report.md`, `examples/report-deep.json`
- Test: manual verification commands below.

**Interfaces:**
- Consumes: the finished CLI.
- Produces: documentation matching the shipped behaviour.

- [ ] **Step 1: Update `docs/design.md`**

Add sections covering, with their provenance: the bits-per-weight table and that
these are llama.cpp *effective* bpw; the `quality_retention` table and that it
comes from published Llama-2-7B perplexity deltas; the anchor formula with the
worked `llama3.1:8b` example (4.9 GB Q4 → predicted 8.63 GB Q8 vs measured
8.5 GB); the assumption that bundled benchmark scores are measured at ~fp16;
the language level/evidence weights; the index-cost formula.

- [ ] **Step 2: Rewrite the README usage section**

Cover `--role`, `--quant`, `--format`, `--kv-quant`, `peyk quant`, `peyk rag`,
`peyk languages`, and the new source/probe flags. Describe the flag change
honestly: `recommend` still has 22 flags — eleven old ones collapsed into seven
while four new capability flags arrived — and the win is that choosing a data
source is now one axis instead of nine, with grouped `--help`. Do not claim a
reduction in flag count.

- [ ] **Step 3: Update the roadmap**

Add M12 (quant ladder), M13 (language levels), M14 (roles + RAG), M15 (CLI
cleanup) to `docs/roadmap.md`, each marked done. Record two open items: removing
the deprecated aliases in the next major, and the deferred `snippet` → `plan
--snippet` merge (raised during design, not approved, deliberately out of scope).

- [ ] **Step 4: Regenerate the example artifacts**

```bash
peyk --json > examples/report.json
peyk --markdown examples/report.md
peyk --probe --json > examples/report-deep.json
peyk rag --languages tr,en --chunks 1000000 --json > examples/rag.json
peyk quant "qwen2.5 14b" --json > examples/quant-ladder.json
```

Update `examples/README.md` to describe the two new artifacts. Verify by hand
that the regenerated reports show a `Lang` column and that at least one model
carries a `~` derived-size marker.

- [ ] **Step 5: Verify and commit**

```bash
pytest -q && ruff check . && mypy src/peyk
peyk --help                    # four groups, no deprecated flags listed
peyk languages tr              # Turkish-capable models with evidence tags
peyk rag --languages tr        # a stack with a co-residency verdict
peyk quant "llama 3.1 8b"      # ladder with measured and derived rows
```

```bash
git add README.md docs/ examples/
git commit -m "Document the quant ladder, roles, language levels and CLI changes"
```

---

## Self-Review

**Spec coverage** — every spec section maps to a task:

| Spec section | Tasks |
|---|---|
| 1. Data model | 1 (enums, variant fields, `index_gb`), 9 (`LanguageSupport`, `lang_evidence`) |
| 2. Quant table + anchored sizing | 2 |
| 2. Availability filtering | 3 |
| 2. KV-cache quantization | 5, 13 (encoder guard) |
| 2. Scoring/selection effect | 6 |
| 2. Registry tags | 4 (`quant_tag`), 8 (caveat) |
| 2. `peyk quant` | 7 |
| 3. Language scoring | 10 |
| 3. Language data + `docs/languages.md` | 11 |
| 3. `Lang` column, `peyk languages` | 12 |
| 4. Catalog additions | 15 |
| 4. Role criteria + index cost | 13, 14 |
| 4. `peyk rag` | 17 |
| 4. `--use-case embedding` cleanup | 16 |
| 5. Flag collapse + aliases | 18 |
| 5. Help groups, config keys | 19 |
| 6. Testing | every task; expected-failure updates called out in 6 and 10 |
| 7. Sequencing | phase order 1→2→3→4 as written |
| 8. Documentation | 20 |

**Naming consistency checked across tasks:** `quant.spec/retention/anchor_factor/derived_size_gb/available_for/expand`; `languages.score` with `LanguageSupport.supports/all_codes`; `estimator.kv_cache_gb(context, variant, kv_bits)`, `index_gb(variant, chunks)`, `estimate_fit(variant, hw, context, kv_bits, chunks)`; `scoring.weights_for(use_case, role)`, `CRITERIA_BY_ROLE`, `index_cost_score`; `engine.recommend(..., kv_bits, only_quant, only_format, role, chunks)`; `rag.select_stack` / `RagStack`. The `chunks` parameter is introduced in Task 13 (estimator) and threaded through `score_variant` → `best_runnable_variant` → `recommend` in Task 17, which is the first task that needs it end-to-end.

**Known cross-task dependency:** Task 9 intentionally leaves `pytest -q` red (`language_score` still expects a list); Task 10 closes it. This is called out in Task 9 Step 4 so an executor does not "fix" it locally.
