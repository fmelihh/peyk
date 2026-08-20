# Accuracy & calibration

peyk's memory and speed numbers are **heuristics**. This page documents how they
are derived and collects real predicted-vs-measured data so the heuristics can be
calibrated over time. Honest calibration is what moves the tool from "interesting"
to "trusted" — contributions from real machines are very welcome.

## What is estimated

- **Memory need** = quantized weight size + KV-cache estimate + runtime overhead.
  - KV cache is GQA-aware and scales with context; MoE models count *active*
    (not total) parameters.
- **Speed (tok/s)** is memory-bandwidth-bound:
  `bandwidth / active_bytes_per_token × backend_factor`.
  - On Linux with `--sudo`, `bandwidth` is **measured** from DIMM speed × channels
    (`dmidecode`); otherwise it is a per-device lookup estimate.

## How to contribute a measurement

1. Pick a model peyk lists as runnable and note peyk's estimate:
   ```bash
   peyk --json | jq '.models[] | select(.model=="Qwen2.5" and .params_b==7)'
   ```
2. Measure the real decode speed. With Ollama:
   ```bash
   ollama run qwen2.5:7b --verbose "Write a paragraph about the sea."
   # read "eval rate: N tokens/s" from the output
   ```
3. Open a PR adding a row to the table below with your hardware, the peyk
   estimate, and the measured rate (and whether you used `--sudo`).

## Predicted vs measured (community-contributed)

> The rows below are **placeholders / illustrative** until real submissions land.
> Do not treat them as validated numbers.

| Machine | Model (quant) | peyk est. tok/s | measured tok/s | `--sudo`? | notes |
|---|---|---:|---:|:--:|---|
| _example_ Ryzen 7 7700 + DDR5-5200 x2 | Qwen2.5 7B (Q4_K_M) | ~11 | _tbd_ | yes | CPU-only |
| _example_ RTX 4090 | Llama 3.1 8B (Q4_K_M) | ~205 | _tbd_ | n/a | full VRAM |
| _example_ Apple M3 | Gemma 2 9B (Q4_K_M) | ~10 | _tbd_ | n/a | unified |

## Known limitations

- Prompt-processing (prefill) speed is not modelled — only decode.
- Backend factors are coarse (llama.cpp vs vLLM vs MLX differ).
- Discrete-GPU partial offload uses a simple GPU/CPU bandwidth blend.
