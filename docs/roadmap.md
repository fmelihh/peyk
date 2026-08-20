# peyk — Feature Roadmap

Derived from a gap analysis against [whichllm](https://github.com/Andyyyy64/whichllm)
(2026-08-20). peyk stays **report-only** (no model download/execution); the one
exception is `snippet`, which only *prints* run commands.

## Guiding identity

Keep and lean into what makes peyk different:

- **Measured** memory bandwidth via native probe (`--sudo`/dmidecode), not a guess.
- **Multi-source cross-check** (Ollama registry + HuggingFace) for real sizes.
- **Language support** as a first-class ranking criterion (incl. Turkish).

Close the ranking/UX gaps below without becoming a model runner.

---

## Milestones (in build order)

### ✅ M1 — CLI subcommands + GPU database (foundation)

Refactor the flat argparse CLI into subcommands while keeping the bare `peyk`
call (no subcommand) behaving as today (`recommend`).

- Subcommands: `recommend` (default), `plan`, `snippet`, `hardware`.
- New `peyk/gpus.py`: a curated GPU spec DB `name -> {vram_gb, bandwidth_gbs,
  vendor}` (fold in and extend the current `bandwidth._NAME_TABLE`). Shared by
  simulation, `plan`, and bandwidth lookup.
- **Done when:** `peyk` unchanged; `peyk hardware` prints the profile; DB covers
  the common NVIDIA/Apple/AMD parts with a fuzzy name lookup + tests.

### ✅ M2 — GPU simulation (`--gpu`) [P0-2]

Recommend for hypothetical hardware without touching the host.

- `--gpu "RTX 4090"`, `--gpu "2x RTX 5090"`, `--gpu "A100 80GB"` (inline VRAM
  override), `--gpu-only` (use full VRAM, no OS reserve).
- Build a synthetic `HardwareProfile` from the GPU DB; skip detection.
- Report banner marks results as **simulated**.
- **Done when:** simulated single/multi-GPU profiles produce correct tiers;
  unknown GPU name → clear error listing close matches.

### ✅ M3 — Reverse lookup (`peyk plan`) [P0-3]

Invert the fit engine: given a model, what does it take to run it?

- `peyk plan "llama 3.3 70b" [--context N] [--quant Q4_K_M]`.
- Resolve the model from the catalog (fuzzy family+size); compute memory need;
  find the smallest GPU class in the DB that FITS, plus the cheapest that is
  TIGHT. Also report CPU/unified RAM needed.
- **Done when:** `plan` prints "needs ≥ X GB → RTX 4090 (24 GB) / A100 (80 GB)"
  with tiers; ambiguous names disambiguate.

### ✅ M4 — `snippet` (report-only-safe) 

`peyk snippet "qwen2.5 7b"` prints ready-to-paste commands to run the chosen
model — it never downloads anything.

- Emits `ollama run <id>` and a minimal `llama.cpp`/`python` snippet for the
  best runnable quant on the current (or `--gpu`) hardware.
- **Done when:** snippet reflects the recommended quant and model_id; `--json`
  variant available.

### ✅ M5 — Evidence-based quality [P0-1] *(highest value, largest)*

Replace the hand-curated `quality_score` proxy with real benchmark evidence and
a confidence signal.

- New `peyk/benchmarks/`: bundled **frozen tier** JSON (Open LLM Leaderboard v2
  + Chatbot Arena ELO snapshots) keyed by model id/family; optional live tier
  (LiveBench / Artificial Analysis) later, best-effort + cached.
- `ModelVariant` gains `quality_source` (`direct|variant|base|interpolated|
  proxy`) and a confidence multiplier; scoring blends normalized benchmark ×
  confidence, falling back to the curated proxy (tagged `proxy`).
- **Repackager guard:** reject cross-family score inheritance when param
  divergence > 2× (protects HF-discovered models).
- Report/JSON expose the confidence tag (e.g. a `Conf` column).
- **Done when:** known models rank by benchmark not size; every quality score is
  tagged with its evidence level; guard unit-tested.

### ✅ M6 — Speed & memory accuracy [P1]

- **MoE active params:** `ModelVariant.active_params_b`; speed uses active (not
  total) bytes/token (fixes Mixtral-class estimates).
- **Partial offload:** for discrete GPUs in the TIGHT tier, model a GPU/CPU
  split and blend bandwidths into the speed estimate.
- **Budget flags:** `--vram-headroom 1.5GB`, `--ram-budget 24GB`,
  `--speed usable|fast` (filter slow models).
- **Done when:** Mixtral speed is realistic; offload speed differs from
  all-in-VRAM; flags change tiers/lists as documented.

### ✅ M7 — Caching [P1]

- TTL cache under `~/.cache/peyk` (respect `$XDG_CACHE_HOME`): models/sizes
  ~6 h, benchmarks ~24 h. Cross-check/discover read cache first.
- **Done when:** a second online run within TTL makes no network calls.

### ✅ M8 — Distribution [P1]

- Publish to **PyPI** (`pip install peyk`, `uvx peyk`, `uv tool install peyk`).
- Add a GitHub Actions CI (pytest matrix) + a release workflow.
- CI (pytest matrix, Ubuntu+macOS, py3.10-3.12) and a tag-triggered PyPI
  release workflow (trusted publishing) are in `.github/workflows/`.
  **Publishing runs on the first `vX.Y.Z` tag** once the PyPI pending publisher
  is registered.

---

### ✅ M9 — Disk-space awareness

Running a model first means downloading it. peyk now detects free disk space
(home dir) and flags models that fit in memory but won't fit on disk.

- `HardwareProfile.disk_free_gb`; shown in the hardware panel and JSON.
- A warning lists runnable models whose GGUF exceeds free disk.

### ✅ M10 — Vision & math task profiles

Close the last whichllm task-profile gap.

- `ModelVariant.modality` ("text"/"vision"); catalog tags Gemma 3, Qwen2.5-VL,
  and Llama 3.2 Vision as multimodal, with benchmark entries.
- `--use-case vision` restricts results to multimodal models; `--use-case math`
  reweights toward quality/context.
- Model resolution prefers a text model on ambiguous ties (explicit "-VL"/vision
  queries still resolve to the vision variant).

### ✅ M11 — Live benchmark tier

Complete the evidence-based-quality vision: frozen snapshots go stale, so add an
opt-in live overlay.

- `peyk/benchmarks/live.py` fetches a JSON benchmark table from
  `PEYK_BENCHMARKS_URL` (or `--benchmarks-url`), cached 24h, fully best-effort.
- `--live-benchmarks` overlays it on the frozen tier; matches are tagged `live`
  (highest trust) and take precedence over `direct`.
- Off by default with no hardcoded endpoint (honest — no fake "live" data).

## Out of scope (deliberate)

- `peyk run` (download + interactive chat) — breaks the report-only identity.
- Serving/deployment, fine-tuning, non-LLM model types.

## Sequencing rationale

M1–M4 are fast, share the GPU DB, and ship visible value without the big data
lift. M5 (benchmarks) is the highest-impact but heaviest, so it follows once the
CLI/DB foundation exists. M6–M8 harden accuracy and reach.
