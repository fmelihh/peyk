# peyk — Design

**Date:** 2026-08-19
**Status:** Approved (autonomous execution granted by user)

## Purpose

A cross-platform Python CLI that inspects the machine it runs on and recommends
the most suitable *current, locally-runnable* LLM models for that hardware. It
produces a **report only** — it does not download, install, or run models.

## Non-goals

- No installation / model execution / benchmarking of real models.
- No web UI or daemon. Single-shot CLI.
- Exact performance guarantees. Speed numbers are labelled estimates.

## Architecture

Six small, independently testable units under `src/peyk/`:

1. **`profiler/`** — Hardware detection. Linux-first, degrades gracefully on
   macOS/Windows. Output: normalized `HardwareProfile`.
2. **`sources/`** — Pluggable model catalog providers implementing a common
   `Source` protocol: `CuratedSource` (bundled JSON, source of truth for
   quality/language metadata), `OllamaSource` (registry manifest sizes,
   best-effort), `HuggingFaceSource` (GGUF repo sizes, best-effort). Results
   are merged/deduped/cross-checked into `ModelCandidate`s.
3. **`estimator.py`** — Memory-fit + rough speed estimate per `ModelVariant`.
4. **`scoring.py`** — Multi-criteria scoring (speed, quality, language,
   context, license).
5. **`report.py`** — Rich CLI tables + JSON/Markdown export.
6. **`cli.py`** — Entry point.

## Data models (`models.py`, pydantic)

`HardwareProfile`: os, arch, cpu cores (phys/logical), cpu flags, ram total/avail,
accelerator (NONE|NVIDIA|APPLE|AMD), vram_total_gb, unified_memory, mem_bandwidth_gbs.

`ModelVariant`: model_id, family, params_b, quant, file_size_gb, context_max,
languages, license, quality_score, source, n_layers?, hidden?, gqa_factor.

`ModelCandidate`: family + params_b + list[ModelVariant].

## Fit estimator (heuristic, documented as rough)

```
mem_need = file_size_gb + kv_cache_gb(context, params) + overhead_gb
kv_cache_gb ≈ context * params_b * 7e-5 * gqa_factor   # fp16 KV, GQA-aware
pool = unified ? ram_available : (accelerator ? vram_total : ram_available)
mem_need <= 0.70 * pool  → FITS   (green)
mem_need <= 0.95 * pool  → TIGHT  (yellow)
else                     → NO_FIT (red)
speed ≈ mem_bandwidth_gbs / active_bytes_per_token * backend_multiplier  # labelled estimate
```

## Scoring criteria (0-100 each)

- **speed** — from estimated tok/s.
- **quality** — curated quality_score proxy (scales with params + known benchmarks).
- **language** — coverage of requested languages (default en; Turkish support valued).
- **context** — from context_max.
- **license** — permissive (apache/mit) high, restricted lower.

Report shows per-criterion top lists **and** a feasibility-tiered table
(FITS / TIGHT / NO_FIT), each candidate reduced to its best runnable quant.

## CLI

`peyk`:
`--use-case chat|coding|summarize|embedding`, `--languages tr,en`,
`--context N`, `--top N`, `--offline`, `--cross-check`, `--discover`,
`--json`, `--markdown FILE`.

## Tech stack

Python 3.10+, `rich`, `psutil`, `httpx`, pydantic v2. Optional: `huggingface_hub`,
`pynvml`. NVIDIA via `pynvml`/`nvidia-smi`, Apple via `sysctl`/`system_profiler`,
AMD via `rocm-smi`. `pytest` for tests.

## Testing

Unit tests with mocked platform calls / API fixtures: estimator math, scoring,
source merge, profiler parsing. Live sources are best-effort and fully mocked.
