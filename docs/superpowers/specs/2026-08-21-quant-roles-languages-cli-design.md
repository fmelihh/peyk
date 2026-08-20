# Design — Quantization ladder, model roles, language levels, CLI simplification

Date: 2026-08-21
Status: approved, ready for planning

## Goal

Four changes to peyk, designed together because all four touch `ModelVariant`:

1. **Quantization is evaluable.** Today 35 of 38 catalog variants are `Q4_K_M`,
   so peyk cannot answer "is Q4 or Q8 right for this box" — a question its own
   README poses. Add a derived quant ladder (GGUF k-quants plus the
   AWQ/GPTQ/FP8/NF4/MLX format families), size-anchored to measured catalog
   sizes, and let it drive both a new `peyk quant` command and the main report.
2. **Embedding and reranker models are first-class.** `--use-case embedding`
   exists but the catalog has no embedding model, so the flag reweights a
   ranking of chat LLMs. Introduce `role`, curate real embedding/reranker
   entries, score them by criteria that apply to them, and add `peyk rag` to
   pick a co-resident embedding + reranker + LLM stack.
3. **Language support becomes real data.** 25 of 38 variants declare
   `["multi"]`, which currently scores a perfect 100 for any requested language.
   Replace it with per-language levels plus an evidence tag, mirroring the
   existing `quality_evidence` design.
4. **The command surface gets a coherent shape.** `recommend` carries 22 flags
   with nine of them describing where data comes from. Collapse those clusters,
   deprecate the old names without breaking them.

Non-goals, unchanged from `docs/roadmap.md`: peyk stays report-only. It does not
download, convert, quantize, or run anything.

## Constraints

- **Offline-first.** Every feature here must produce a full answer with no
  network. The quant ladder is computed, not fetched; language levels and
  embedding metadata are bundled in the catalog.
- **Auditable.** Every number peyk prints must be traceable to either a measured
  input or a documented heuristic, and labelled as to which. This design adds
  two new evidence axes (`size_evidence`, `lang_evidence`) for exactly that
  reason.
- **Backward compatible.** Third-party catalogs (`--catalog-url`) and
  HF-discovered variants use the current schema. All new fields carry defaults
  and all changed fields accept their old shape.

---

## 1. Data model

`src/peyk/models.py` — `ModelVariant` gains:

```python
role: Role = Role.LLM                        # llm | embedding | reranker
languages: LanguageSupport                   # was list[str]
lang_evidence: LangEvidence = LangEvidence.UNKNOWN
quant_format: QuantFormat = QuantFormat.GGUF # gguf | awq | gptq | fp8 | nf4 | mlx | fp16
size_evidence: SizeEvidence = SizeEvidence.MEASURED  # measured | derived
quant_tag: str | None = None                 # conventional registry suffix, e.g. "q6_K"
embed_dim: int | None = None                 # role=embedding only
```

`LanguageSupport` is a small pydantic model:

```python
class LanguageSupport(BaseModel):
    native: list[str] = []
    good: list[str] = []
    partial: list[str] = []
    catchall: Level | None = None   # level for any code not listed above
```

with a `model_validator(mode="before")` accepting the legacy shape:

- `list[str]` containing `"multi"` → every other listed code goes to `good`, and
  `catchall` is set to `partial`, so a language the entry never names resolves
  to `partial` instead of "unsupported". This is what makes an unaudited
  `"multi"` claim stop scoring 100 without making it score 0.
- any other `list[str]` → all codes land in `good`, `catchall` stays `None`
  (an explicit list is a closed list).

A `supports(code) -> Level | None` helper is the single read path; nothing
outside the model inspects the three lists directly.

`FitResult` gains `index_gb: float = 0.0` (vector-index cost, embedding role
only, zero elsewhere) so the JSON report stays one flat shape across roles.

## 2. Phase 1 — Quantization ladder (`src/peyk/quant.py`)

### The table

A frozen `QuantSpec` per entry: `name`, `bits_per_weight`, `quality_retention`,
`format`, `requires`.

| Quant | bpw | Retention | Format | Requires |
|---|---|---|---|---|
| F16 | 16.0 | 1.000 | gguf | — |
| Q8_0 | 8.5 | 0.999 | gguf | — |
| Q6_K | 6.56 | 0.997 | gguf | — |
| Q5_K_M | 5.67 | 0.993 | gguf | — |
| Q5_K_S | 5.52 | 0.991 | gguf | — |
| Q4_K_M | 4.83 | 0.985 | gguf | — |
| Q4_K_S | 4.57 | 0.981 | gguf | — |
| IQ4_XS | 4.25 | 0.978 | gguf | — |
| Q3_K_M | 3.89 | 0.955 | gguf | — |
| IQ3_M | 3.66 | 0.945 | gguf | — |
| Q3_K_S | 3.50 | 0.930 | gguf | — |
| IQ2_M | 2.70 | 0.900 | gguf | — |
| Q2_K | 2.63 | 0.880 | gguf | — |
| FP16 | 16.0 | 1.000 | fp16 | GPU |
| FP8 | 8.0 | 0.998 | fp8 | NVIDIA, compute cap ≥ 8.9 |
| AWQ-4 | 4.15 | 0.982 | awq | NVIDIA, compute cap ≥ 7.5 |
| GPTQ-4 | 4.15 | 0.978 | gptq | NVIDIA, compute cap ≥ 6.1 |
| NF4 | 4.50 | 0.975 | nf4 | CUDA |
| MLX-8 | 8.5 | 0.999 | mlx | Apple Silicon |
| MLX-4 | 4.5 | 0.980 | mlx | Apple Silicon |

`bits_per_weight` are llama.cpp effective bpw (they include the
`token_embd`/`output` tensor overrides, which is why Q4_K_M is 4.83 and not
4.5). `quality_retention` is derived from published perplexity deltas on
Llama-2-7B, expressed as a multiplier so it composes with the existing quality
score. Both tables live in one module with a docstring citing their provenance,
and are documented in `docs/design.md` as heuristics.

### Sizing: anchor to measured, don't guess

Raw size is `params_b * bpw / 8` GB, but that ignores per-family vocabulary and
embedding overhead. So the ladder is **anchored**:

```
k = measured_size_gb / (params_b * bpw(measured_quant) / 8)
derived_size_gb(q) = params_b * bpw(q) / 8 * k
```

When a family+size has several measured catalog rows, `k` is their mean.
Verification against existing catalog data: `llama3.1:8b` Q4_K_M is 4.9 GB →
k = 1.015 → predicted Q8_0 = 8.63 GB against the catalog's measured 8.5 GB
(1.5% error). A regression test pins this.

Ladder entries that exist in the catalog keep their measured size and are tagged
`size_evidence=measured`; the rest are tagged `derived`. The report's `Source`
column shows the distinction.

### Availability filtering

`quant.available_for(hw) -> list[QuantSpec]` filters by `requires`:

- `Accelerator.NONE` → GGUF only.
- `Accelerator.APPLE` → GGUF + MLX.
- `Accelerator.NVIDIA` → GGUF + AWQ/GPTQ/NF4/INT8/FP16, plus FP8 when
  `gpu_compute_cap >= 8.9`. Compute capability is already on `HardwareProfile`;
  when it is `None` (simulated GPUs, driverless hosts) fall back to the GPU DB
  entry, and if that is unknown too, omit FP8 rather than assume it.
- `Accelerator.AMD` → GGUF + GPTQ + FP16.

### KV-cache quantization

`estimator.kv_cache_gb(context, variant, kv_bits=16)` — KV cache scales by
`kv_bits / 16`. Exposed as `--kv-quant f16|q8_0|q4_0` (16/8/4 bits). This is
often the deciding factor for long-context fit, and it is a separate axis from
weight quantization.

Correctness note: embedding and reranker models are encoders with no
autoregressive cache, so `kv_cache_gb` returns 0 when `role != llm`.

### Effect on scoring and selection

`engine.recommend` expands each candidate into its ladder before scoring:
`quant.expand(candidate, hw) -> list[ModelVariant]`, deduplicated by quant with
measured rows winning over derived. `scoring.quality_score` becomes
`benchmarks.evaluate(v).effective * quant.retention(v.quant)`.

The retention multiplier is what stops the selector from always taking the
cheapest quant: without it, every model's best-scoring variant would be the
fastest one that fits. Documented assumption, stated in `docs/design.md`: the
bundled benchmark snapshot is treated as measured at ~fp16.

This is a deliberate behaviour change. On a 16 GB machine, `llama3.1:8b` will be
recommended at Q6_K rather than Q4_K_M; on an 8 GB box, `qwen2.5:14b` appears as
`IQ3_M / TIGHT` where it previously read `WON'T FIT`.

### Registry tags

Derived variants keep the anchor's `model_id` and carry `quant_tag` (e.g.
`q6_K`). `snippet.build_snippets` composes `f"{model_id}-{quant_tag}"` for the
Ollama command **and prints a caveat** when `size_evidence == derived`, because
a registry tag for that exact quant may not exist. `--online` cross-check
verifies the tag against the Ollama registry and clears the caveat when it
resolves. Printing an unverified `ollama run` line without this caveat would be
a correctness bug, not a cosmetic one.

### New command

```
peyk quant "qwen2.5 14b" [--context N] [--format gguf|awq|mlx|...] [--json]
```

Columns: Quant, Format, Size, Mem@ctx, t/s, ΔQuality, Tier — sorted by bpw
descending, with the variant `recommend` would pick starred.

## 3. Phase 2 — Language levels (`src/peyk/languages.py`)

Sequenced before the embedding work so the ~17 new catalog rows are authored
once, in the new schema.

### Scoring

```
level weight:    native 100 · good 80 · partial 45 · absent 0
evidence factor: benchmark 1.00 · stated 0.95 · inferred 0.80 · unknown 0.60
language_score = mean(level_weight(lang) for lang in requested) * evidence_factor
```

This deliberately mirrors `benchmarks.QualityEvidence`: same shape, same
confidence-discount idea, so the report tells one consistent story about what
peyk measured versus what it was told.

### Data

All 38 existing variants plus the 17 new ones are curated by family (19 + new
families), sourced from official model cards:

- `benchmark` — a published multilingual benchmark covers the language
  (Global-MMLU, Belebele, MMMLU, MIRACL for embeddings).
- `stated` — the model card names the language as supported.
- `inferred` — no claim, but strong related-language or tokenizer evidence.
- `unknown` — nothing; the default for HF-discovered variants.

`docs/languages.md` records, per family, where each level came from — the
curation is a judgement call and must be reviewable.

### Surfaces

- **`Lang` column** in the feasibility table, one glyph per requested language:
  `●` native, `◐` good, `○` partial, `·` absent. `--plain` degrades to
  `tr:good`.
- **`peyk languages`** — no argument: every language in the catalog with a model
  count per level. With an argument (`peyk languages tr`): models supporting it,
  ordered by level then quality, showing the evidence tag.

`--languages` stays a ranking signal, not a filter (`audit --require-language`
remains the hard gate).

## 4. Phase 3 — Roles: embedding and reranker

### Catalog additions

**Embedding (11):** bge-m3, multilingual-e5-large, multilingual-e5-base,
Qwen3-Embedding-0.6B / 4B / 8B, nomic-embed-text-v1.5, gte-multilingual-base,
embeddinggemma-300m, jina-embeddings-v3, snowflake-arctic-embed-l-v2.0,
all-MiniLM-L6-v2.

**Reranker (5):** bge-reranker-v2-m3, Qwen3-Reranker-0.6B / 4B,
jina-reranker-v2-base-multilingual, mxbai-rerank-base-v2.

Multilingual coverage (Turkish in particular) is weighted deliberately in this
selection, consistent with peyk's language-aware positioning. Licenses are
recorded honestly, including the non-commercial Jina entries — they score low on
license and that is the correct signal.

### Role-specific criteria

`engine.CRITERIA` becomes a per-role lookup:

| Role | Criteria |
|---|---|
| llm | speed, quality, language, context, license *(unchanged)* |
| embedding | quality, language, context, license, **index_cost** |
| reranker | quality, language, context, license, speed |

`index_cost` is peyk-specific and useful: an embedding model's real cost is
usually its vector index, not its weights.

```
index_gb = chunks * embed_dim * 4 bytes / 1e9        # fp32
```

At `dim 1024`, one million chunks is 4.1 GB — larger than every embedding model
in the catalog. Scored inversely (smaller index → higher score), reported as an
absolute figure alongside.

Runtime overhead differs by role: `RUNTIME_OVERHEAD_GB` stays 0.9 for LLMs and
drops to 0.3 for embedding/reranker, which run far lighter serving stacks.

### `peyk rag`

```
peyk rag [--languages tr,en] [--context 8192] [--chunks 1000000] [--no-reranker]
```

Selects the highest-scoring **co-resident** triple, not three independent
winners: the objective is the best combined score subject to
`sum(mem_need) + index_gb <= memory_pool_gb`. The search space is small (models
× models × models over a catalog of tens), so an exhaustive search over the
top-N of each role is sufficient and deterministic — no heuristic solver.

Output: one row per role, the memory total against the pool, and an explicit
verdict on whether all three fit simultaneously.

### Related cleanup

`--use-case embedding` conflates a scoring profile with a model type. It becomes
a deprecated alias for `--role embedding` (warns, still works, removed next
major).

## 5. Phase 4 — CLI simplification

| Before | After |
|---|---|
| `--offline` (default behaviour) | *(default; flag kept as hidden alias)* |
| `--cross-check`, `--discover`, `--live-benchmarks` | `--online[=discover\|all]` |
| `--catalog-url URL` | `--source URL` |
| `--benchmarks-url URL` | `--benchmarks URL` |
| `--no-cache` | `--refresh` |
| `--deep`, `--sudo` | `--probe[=deep]` |
| `--vram-headroom 1.5GB`, `--ram-budget 24GB` | `--reserve 1.5GB`, `--memory 24GB` |

`--online` takes an optional value: bare `--online` means live size cross-check
plus the live benchmark tier; `--online=discover` adds HuggingFace discovery;
`--online=all` enables everything. Absent the flag, peyk is offline — which is
already today's default and stays the identity of the tool.

Every old flag survives as an `argparse.SUPPRESS`-ed alias that emits one
stderr deprecation line and maps onto the new option. Removal is scheduled for
the next major version and recorded in the roadmap.

`--help` is organised into four argument groups: **Hardware**, **Sources**,
**Filters**, **Output**.

**Honest accounting of the result.** `recommend` goes from 22 flags to 22:
eleven old flags collapse into seven, and four new capability flags
(`--role`, `--quant`, `--format`, `--kv-quant`) arrive from phases 1–3. The win
is not flag count — it is that choosing a data source stops being a nine-flag
puzzle and becomes one axis, and that `--help` is skimmable. The README will
describe it in those terms rather than claiming a reduction that did not happen.

Subcommands grow from five to eight (`quant`, `rag`, `languages`). Folding
`snippet` into `plan --snippet` would offset this and the two commands do
overlap, but it was raised and not approved, so it stays out of scope and is
recorded in the roadmap as an open question.

New `.peyk.json` keys: `role`, `quant`, `format`, `kv_quant`. `config.CONFIG_KEYS`
is extended accordingly.

## 6. Testing

Test-driven, one file per phase, matching the existing `tests/` layout.

- `test_quant.py` — bpw arithmetic; the anchor formula against the known
  `llama3.1:8b` Q4/Q8 pair; `available_for` across all four accelerator types
  including the FP8 compute-cap boundary and the unknown-capability fallback;
  retention changing which quant wins on a large-memory profile; KV-quant
  scaling; derived-tag caveat appearing in `snippet`.
- `test_languages.py` — level × evidence scoring; legacy `list[str]` and
  `["multi"]` inputs producing `good` / `partial` respectively; `peyk languages`
  with and without an argument.
- `test_roles.py`, `test_rag.py` — per-role criteria selection; zero KV cache
  for encoders; `index_gb` arithmetic; the co-residency constraint rejecting a
  triple that individually fits but jointly does not.
- `test_cli_flags.py` — every deprecated alias produces the same parsed result
  as its replacement, and emits exactly one warning.

Existing tests expected to need updating, because the behaviour they pin is
intentionally changing: `test_estimator.py` (KV signature), `test_scoring.py`
(retention factor, language scoring), `test_engine_and_cli.py` (ladder
expansion, new flags). These are updates to match new intent, not fixes to
accidental breakage — the distinction gets called out in the commits.

## 7. Sequencing

`1 → 2 → 3 → 4` as numbered above: quant ladder, language levels, roles/RAG,
CLI. Languages precede roles so the new embedding rows are authored once in the
final schema. CLI simplification is last because the preceding phases each add
flags, and the surface should be tidied with all of them visible.

Each phase is independently testable and independently committable.

## 8. Documentation

- `README.md` — usage section rewritten around the new surface.
- `docs/design.md` — bpw and retention tables with provenance; the anchor
  formula; the fp16-benchmark assumption; the language scoring formula.
- `docs/languages.md` — new; per-family curation sources.
- `docs/roadmap.md` — M12–M15; the deferred `snippet`/`plan` merge.
- `examples/` — regenerated artifacts.
