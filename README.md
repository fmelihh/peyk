<p align="center">
  <img src="assets/logo.svg" alt="peyk" width="440">
</p>

<p align="center">
  <b>Offline, auditable, hardware-measured model selection for local LLMs.</b>
</p>

<p align="center">
  <a href="https://github.com/fmelihh/peyk/actions/workflows/ci.yml"><img src="https://github.com/fmelihh/peyk/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/peyk/"><img src="https://img.shields.io/pypi/v/peyk" alt="PyPI"></a>
  <img src="https://img.shields.io/pypi/pyversions/peyk" alt="Python versions">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

`peyk` inspects the machine it runs on and tells you which *current,
locally-runnable* LLM models that machine can actually run — ranked across
speed, quality, language support, context length, and license. It produces a
**report only**: it never downloads, installs, or runs a model.

## Where peyk is different

There are plenty of "will it run" tools (whichllm, LM Studio's fit badge, HF's
Model Memory Calculator, various VRAM websites). peyk is built for the corner
they don't cover well — **offline, regulated, and CPU/edge deployments**:

- 🔬 **Measured, not guessed hardware.** With `--sudo` on Linux, peyk reads real
  DIMM speed and channel count via `dmidecode` and computes a *measured* memory
  bandwidth — the single biggest driver of CPU/edge token/s. It also surfaces
  NUMA topology, swap, and GPU driver. Others stop at psutil/nvidia-smi.
- 🔒 **Air-gapped by design.** The catalog is bundled and deterministic; peyk
  gives a full answer with **no network**. Web calculators can't run in a closed
  network; whichllm leans on the live HF API.
- 🛡 **Policy audits (`peyk audit`).** Gate model choices in CI against a license
  allow-list, size cap, required languages, and a quality floor — exit code for
  pipelines. Useful for compliance/on-prem review, not just picking a chat model.
- 🌐 **Language-aware ranking.** `--languages tr,en` ranks by language support
  (Turkish included) — no other tool here does this.
- 🖥 **Hardware simulation + reverse lookup.** `--gpu "2x RTX 5090"` and
  `peyk plan "llama 3.3 70b"` without owning the hardware.

If you just want a chat model on a gaming GPU with internet, LM Studio's badge is
enough. peyk earns its place on **CPU boxes, air-gapped racks, and CI gates**.

## Why the name?

In the Ottoman court, a **peyk** was a class of royal courier who ran ahead on
foot to carry news and messages between the palace, the army, and the provinces —
fast, light, and always arriving before the caravan. This tool plays the same
role in inference engineering: before you commit to downloading a 40 GB model or
provisioning a GPU box, `peyk` runs ahead and delivers the news of *what will
actually run here* — so you don't find out the hard way.

## The problem it solves

Picking a local model is a hardware-fit problem disguised as a model-choice
problem. Whether you're doing **on-prem / air-gapped inference**, **edge
deployment**, **privacy-sensitive workloads**, or just running models on a
laptop, the same questions come up every time:

- Will a 14B model fit in my VRAM once the KV cache is included?
- Is Q4 or Q8 the right quantization for this box?
- Which of today's models is the best *speed / quality / language* trade-off for
  *this* machine — not a benchmark machine?

Answering these by hand means reading model cards, guessing memory footprints,
and trial-and-error downloads. `peyk` turns that into one command.

## Installation

`peyk` needs **Python 3.10+**. It runs on Linux (primary target), macOS, and
Windows.

### Quickstart

```bash
uvx peyk                     # run without installing (one-off)
uv tool install peyk         # persistent install
pip install peyk             # classic pip
```

Then just run `peyk`. For hacking on the source, install from a clone instead:

### Linux (recommended)

```bash
git clone https://github.com/fmelihh/peyk.git
cd peyk

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
# NVIDIA GPU box? add the optional accelerator dependency:
pip install -e ".[nvidia]"
```

GPU detection uses standard vendor tools that are already present on a properly
configured host — no extra setup needed:

- **NVIDIA:** `nvidia-smi` (from the NVIDIA driver). Multiple GPUs are detected
  and their VRAM is aggregated.
- **AMD (ROCm):** `rocm-smi`.
- **CPU-only:** works out of the box; nothing to install.

### macOS (Apple Silicon / Intel)

```bash
git clone https://github.com/fmelihh/peyk.git
cd peyk
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Apple Silicon, unified memory is detected automatically and treated as a
shared weight + KV-cache pool.

### Windows (PowerShell)

```powershell
git clone https://github.com/fmelihh/peyk.git
cd peyk
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### Run without activating a virtualenv

```bash
pipx install .          # from a clone
# or, for a one-off:
python -m peyk --help
```

## Usage

```bash
peyk                              # default report (offline bundled catalog)
peyk --use-case coding            # weight the ranking toward a use case
peyk --languages tr,en            # favor models that support these languages
peyk --context 32768              # size for a target context length
peyk --top 10                     # show more models per criterion
peyk --speed fast                 # hide slow models (usable≥4, fast≥10 tok/s)
peyk --vram-headroom 1.5GB        # keep memory free (safety margin)
peyk --ram-budget 24GB            # cap the usable pool
peyk --cross-check                # verify sizes live via Ollama + HuggingFace
peyk --discover                   # pull in trending GGUF models from HuggingFace
peyk --deep                       # native probe: RAM type/speed, exact chip
sudo peyk --sudo                  # deepest probe: measured memory bandwidth
peyk --json                       # machine-readable output (for CI / scripts)
peyk --markdown report.md         # write a shareable Markdown report
```

`--use-case` accepts `chat`, `coding`, `summarize`, `embedding`, `vision`, or
`math`. `vision` restricts the results to multimodal models.

### Subcommands

A bare `peyk` runs `recommend`. There are three more:

```bash
# Simulate hardware you don't have yet (before buying a GPU)
peyk --gpu "RTX 4090"             # single card
peyk --gpu "2x RTX 5090"          # multi-GPU (VRAM aggregated)
peyk --gpu "A100 80GB" --gpu-only # full VRAM, no OS reserve

# Reverse lookup: what does it take to run a given model?
peyk plan "llama 3.3 70b"         # -> memory needed + cheapest GPU that fits
peyk plan "qwen2.5 32b" --context 32768

# Print ready-to-run commands for a model (no download)
peyk snippet "qwen2.5 7b"         # -> ollama / python / llama.cpp commands

# Show just the detected (or simulated) hardware
peyk hardware --deep
peyk hardware --gpu "2x A100 80GB"

# Policy gate for CI / on-prem review (exit 0 = compliant model available)
peyk audit --allow-license apache-2.0,mit --max-params 32 --require-language tr
peyk audit --policy policy.json --json
```

A policy file (`--policy policy.json`):

```json
{
  "max_params_b": 32,
  "allow_licenses": ["apache-2.0", "mit"],
  "require_languages": ["tr", "en"],
  "min_quality": 60,
  "require_fit": true
}
```

### Deep hardware probe (`--deep` / `--sudo`)

By default `peyk` reads hardware with portable Python (psutil) plus vendor CLIs.
For a sharper picture it ships native probe scripts that call system tools
directly:

| Platform | Script | Extra tools |
|---|---|---|
| Linux   | `collect_linux.sh`   | `lscpu`, `nvidia-smi`/`rocm-smi`, `dmidecode` (root) |
| macOS   | `collect_macos.sh`   | `sysctl`, `system_profiler` |
| Windows | `collect_windows.ps1`| CIM/WMI, `nvidia-smi` |

- **`--deep`** runs the script for your OS and adds RAM type/speed, DIMM count,
  and the exact chip name (e.g. `Apple M3 Max`, which fixes the bandwidth model).
- **`--sudo`** (Linux) additionally runs `dmidecode`, turning the memory
  bandwidth from a table *estimate* into a *measured* value derived from your
  actual DIMM speed and channel count — the single biggest input to the token/s
  estimate for CPU inference. Run as `sudo peyk --sudo`, or `peyk --sudo` and
  enter your password at the prompt.

The scripts are bundled with the package and are pure read-only inspection;
`peyk` still works without them (it just falls back to the baseline profile).

### Example (CPU + JSON, e.g. inside CI)

```bash
peyk --json --use-case chat | jq '.models[] | select(.tier=="FITS") | .model' | head
```

## What the report shows

- **Hardware Profile** — OS/arch, CPU, RAM, free disk, accelerator, VRAM
  (aggregated across GPUs), the usable memory pool used for sizing, and an
  estimated memory bandwidth. Models that fit in memory but exceed free disk
  (you still have to download them) are flagged.
- **Feasibility table** — every model reduced to its best runnable quantization,
  grouped into `RUNS WELL` / `TIGHT` / `WON'T FIT`, with estimated memory,
  estimated speed, an overall score, and a **Source** column
  (`curated` / `ollama` / `HF✓` / `HF discover`).
- **Top-by-criterion** tables — best models for Speed, Quality, Language,
  Context, and License.

## Examples

Ready-to-run recipes and real sample outputs live in
[`examples/`](examples/) ([`examples/README.md`](examples/README.md)).

```bash
# Best all-round model for chat, favoring Turkish + English
peyk --use-case chat --languages tr,en

# Coding model sized for a 32k context window
peyk --use-case coding --context 32768

# GPU server: aggregate multi-GPU VRAM and verify sizes live
peyk --cross-check --top 10

# CPU box: measured memory bandwidth via dmidecode
sudo peyk --sudo
```

Compose it into scripts with `--json`:

```bash
# Every model that fits comfortably, best first
peyk --json | jq -r '.models[] | select(.tier=="FITS") | "\(.overall)\t\(.model) \(.params_b)B \(.quant)"' | sort -rn

# The single top recommendation
peyk --json | jq -r '[.models[] | select(.tier!="NO_FIT")] | max_by(.overall) | .model_id'
```

Sample artifacts (generated on an Apple M3 / 16 GB laptop):
[`report.json`](examples/report.json) ·
[`report.md`](examples/report.md) ·
[`report-deep.json`](examples/report-deep.json).

## Data sources

1. **Curated catalog** (bundled, offline, deterministic) — the source of truth
   for quality and language metadata:
   `src/peyk/sources/data/catalog.json`.
2. **Ollama registry** (`--cross-check`) — refreshes on-disk sizes from the
   public Ollama registry manifests.
3. **HuggingFace** (`--cross-check`, `--discover`) — verifies GGUF sizes and,
   with `--discover`, surfaces trending GGUF text-generation models that aren't
   in the curated set (marked `HF discover`; non-LLM GGUF such as ASR/embedding
   are filtered out).

Live sources are **best-effort**: if the network is unavailable, `peyk` falls
back to the curated catalog and still produces a report.

## Catalog freshness

The model world ages in months, so a stale catalog silently gives bad advice.
peyk guards against that:

- Every report shows **when the catalog was last updated** ("updated 2026-08-20
  (1 days ago)") and warns past 45 days.
- The catalog can be pulled from a **versioned remote endpoint** without a code
  release: `peyk --catalog-url https://…/catalog.json` (or `PEYK_CATALOG_URL`),
  cached locally with offline fallback.
- `--cross-check` / `--discover` refresh sizes and surface new models live.
- A **weekly GitHub Action** (`scripts/refresh_catalog.py`) re-verifies file sizes
  against the Ollama registry and opens a rolling PR, so the bundled catalog stays
  current without manual effort (new models / quality are still curated by hand).

## Accuracy & calibration

peyk is honest that memory and speed are **heuristics** — but the estimates are
grounded and improvable:

- Memory = quantized weights + a GQA/MoE-aware KV-cache estimate + runtime
  overhead. MoE models use *active* (not total) params.
- Speed is memory-bandwidth-bound. On Linux with `--sudo`, bandwidth is
  **measured** from real DIMM specs rather than a lookup — the biggest accuracy
  win for CPU/edge boxes.

Real predicted-vs-measured tok/s numbers (community-contributed) live in
[`docs/accuracy.md`](docs/accuracy.md). Contributions from real machines are
welcome — that's how this graduates from "interesting" to "trusted".

## How it works

1. `profiler/` normalizes the host into a `HardwareProfile` (baseline via psutil;
   optional native probe scripts enrich it under `--deep`/`--sudo`).
2. `sources/` assembles the catalog (bundled JSON + optional live cross-check /
   discovery).
3. `estimator.py` computes per-variant memory need (weights + KV cache +
   overhead) and a rough tokens/sec estimate.
4. `scoring.py` scores each model 0–100 on five criteria, weighted by use case.
   Quality is **evidence-based**: a bundled benchmark snapshot
   (`peyk/benchmarks/`) is matched `direct` → `interpolated` → `family` → `proxy`
   and discounted by confidence, so unknown/repackaged models can't inherit a
   score they haven't earned. An optional **live tier** (`--live-benchmarks`
   with `PEYK_BENCHMARKS_URL`) overlays fresh scores tagged `live`.
5. `report.py` renders to terminal / JSON / Markdown.

> **Note:** memory and speed figures are deliberately coarse heuristics; real
> results depend on the backend, quantization, and context length. For
> `HF discover` models, the quality score is estimated from parameter count
> only (there is no benchmark signal). See [`docs/design.md`](docs/design.md).

## Updating the catalog

The model list lives in `src/peyk/sources/data/catalog.json`. To add a model,
append a variant row (params, quant, file size, context, languages, license,
quality proxy). `--cross-check` will verify the sizes against live registries.

## Development

```bash
pip install -e ".[dev]"
pytest -q          # tests
ruff check .       # lint
mypy src/peyk      # types
```

## Star History

<a href="https://star-history.com/#fmelihh/peyk&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=fmelihh/peyk&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=fmelihh/peyk&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=fmelihh/peyk&type=Date" width="600" />
  </picture>
</a>

## License

MIT — see [`LICENSE`](LICENSE).
