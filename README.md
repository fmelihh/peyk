# peyk

**Right-size local LLMs to your hardware.** `peyk` inspects the machine it runs
on and tells you which *current, locally-runnable* LLM models that machine can
actually run — ranked across speed, quality, language support, context length,
and license. It produces a **report only**: it never downloads, installs, or
runs a model.

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
peyk --cross-check                # verify sizes live via Ollama + HuggingFace
peyk --discover                   # pull in trending GGUF models from HuggingFace
peyk --deep                       # native probe: RAM type/speed, exact chip
sudo peyk --sudo                  # deepest probe: measured memory bandwidth
peyk --json                       # machine-readable output (for CI / scripts)
peyk --markdown report.md         # write a shareable Markdown report
```

`--use-case` accepts `chat`, `coding`, `summarize`, or `embedding`.

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

- **Hardware Profile** — OS/arch, CPU, RAM, accelerator, VRAM (aggregated across
  GPUs), the usable memory pool used for sizing, and an estimated memory bandwidth.
- **Feasibility table** — every model reduced to its best runnable quantization,
  grouped into `RUNS WELL` / `TIGHT` / `WON'T FIT`, with estimated memory,
  estimated speed, an overall score, and a **Source** column
  (`curated` / `ollama` / `HF✓` / `HF discover`).
- **Top-by-criterion** tables — best models for Speed, Quality, Language,
  Context, and License.

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

## How it works

1. `profiler/` normalizes the host into a `HardwareProfile` (baseline via psutil;
   optional native probe scripts enrich it under `--deep`/`--sudo`).
2. `sources/` assembles the catalog (bundled JSON + optional live cross-check /
   discovery).
3. `estimator.py` computes per-variant memory need (weights + KV cache +
   overhead) and a rough tokens/sec estimate.
4. `scoring.py` scores each model 0–100 on five criteria, weighted by use case.
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
pytest -q
```

## License

MIT — see [`LICENSE`](LICENSE).
