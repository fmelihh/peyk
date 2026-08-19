# peyk — LLM model recommendation

## Hardware
- OS/Arch: Darwin / arm64
- RAM: 6.5 / 16.0 GB
- Accelerator: APPLE Apple M3
- Usable memory pool: 12.8 GB (~100 GB/s)

## Feasibility

| Status | Model | Params | Quant | Memory | Speed (est.) | Score | Source |
|---|---|---:|---|---:|---:|---:|---|
| RUNS WELL | Llama 3.2 | 1B | Q4_K_M | 1.8 GB | ~69 tok/s | 82 | curated |
| RUNS WELL | Nomic Embed | 0.14B | F16 | 1.2 GB | ~183 tok/s | 80 | curated |
| RUNS WELL | mxbai Embed | 0.33B | F16 | 1.6 GB | ~79 tok/s | 79 | curated |
| RUNS WELL | Qwen2.5 | 0.5B | Q4_K_M | 1.4 GB | ~138 tok/s | 78 | curated |
| RUNS WELL | Llama 3.2 | 3B | Q4_K_M | 3.3 GB | ~28 tok/s | 73 | curated |
| RUNS WELL | Phi-3.5-mini | 3.8B | Q4_K_M | 3.8 GB | ~23 tok/s | 73 | curated |
| RUNS WELL | Qwen2.5 | 3B | Q4_K_M | 3.3 GB | ~28 tok/s | 73 | curated |
| RUNS WELL | DeepSeek-R1-Distill | 8B | Q4_K_M | 7.0 GB | ~11 tok/s | 72 | curated |
| RUNS WELL | DeepSeek-R1-Distill | 7B | Q4_K_M | 6.6 GB | ~12 tok/s | 72 | curated |
| RUNS WELL | Gemma 2 | 2B | Q4_K_M | 2.8 GB | ~34 tok/s | 72 | curated |
| RUNS WELL | Qwen3 | 8B | Q4_K_M | 7.2 GB | ~11 tok/s | 72 | curated |
| RUNS WELL | Gemma 3 | 4B | Q4_K_M | 4.8 GB | ~17 tok/s | 71 | curated |
| RUNS WELL | Llama 3.1 | 8B | Q4_K_M | 7.0 GB | ~11 tok/s | 71 | curated |
| RUNS WELL | Qwen2.5 | 7B | Q4_K_M | 6.6 GB | ~12 tok/s | 71 | curated |
| RUNS WELL | Qwen2.5-Coder | 7B | Q4_K_M | 6.6 GB | ~12 tok/s | 70 | curated |
| RUNS WELL | Qwen2.5-VL | 7B | Q4_K_M | 6.9 GB | ~11 tok/s | 70 | curated |
| RUNS WELL | Mistral | 7B | Q4_K_M | 6.3 GB | ~12 tok/s | 69 | curated |
| RUNS WELL | Gemma 2 | 9B | Q4_K_M | 8.0 GB | ~10 tok/s | 68 | curated |
| RUNS WELL | Aya Expanse | 8B | Q4_K_M | 7.2 GB | ~11 tok/s | 60 | curated |
| TIGHT | Gemma 3 | 12B | Q4_K_M | 10.7 GB | ~7 tok/s | 72 | curated |
| TIGHT | Qwen2.5 | 14B | Q4_K_M | 11.9 GB | ~6 tok/s | 70 | curated |
| TIGHT | Phi-4 | 14B | Q4_K_M | 12.0 GB | ~6 tok/s | 70 | curated |
| TIGHT | Llama 3.2 Vision | 11B | Q4_K_M | 10.4 GB | ~7 tok/s | 69 | curated |
| WON'T FIT | DeepSeek-R1-Distill | 32B | Q4_K_M | 25.5 GB | ~3 tok/s | 72 | curated |
| WON'T FIT | DeepSeek-R1-Distill | 70B | Q4_K_M | 53.9 GB | ~1 tok/s | 72 | curated |
| WON'T FIT | Llama 3.3 | 70B | Q4_K_M | 53.9 GB | ~1 tok/s | 72 | curated |
| WON'T FIT | Gemma 3 | 27B | Q4_K_M | 21.8 GB | ~3 tok/s | 72 | curated |
| WON'T FIT | Qwen3 | 32B | Q4_K_M | 25.5 GB | ~3 tok/s | 72 | curated |
| WON'T FIT | Llama 3.2 Vision | 90B | Q4_K_M | 68.8 GB | ~1 tok/s | 71 | curated |
| WON'T FIT | Mixtral | 47B | Q4_K_M | 34.0 GB | ~8 tok/s | 71 | curated |
| WON'T FIT | Qwen2.5 | 32B | Q4_K_M | 25.5 GB | ~3 tok/s | 71 | curated |
| WON'T FIT | Qwen2.5-Coder | 32B | Q4_K_M | 25.5 GB | ~3 tok/s | 71 | curated |
| WON'T FIT | Mistral Small 3 | 24B | Q4_K_M | 18.6 GB | ~4 tok/s | 70 | curated |
| WON'T FIT | Gemma 2 | 27B | Q4_K_M | 21.4 GB | ~3 tok/s | 69 | curated |
| WON'T FIT | Qwen2.5 | 72B | Q4_K_M | 58.2 GB | ~1 tok/s | 68 | curated |
| WON'T FIT | Qwen2.5-VL | 72B | Q4_K_M | 58.2 GB | ~1 tok/s | 68 | curated |
| WON'T FIT | Aya Expanse | 32B | Q4_K_M | 25.5 GB | ~3 tok/s | 64 | curated |

### Top by Speed
1. Llama 3.2 1B — 100
2. Qwen2.5 0.5B — 100
3. Nomic Embed 0.14B — 100
4. mxbai Embed 0.33B — 100
5. Gemma 2 2B — 57

### Top by Quality
1. Phi-4 14B — 81
2. Qwen2.5 14B — 80
3. Gemma 3 12B — 79
4. Qwen3 8B — 77
5. DeepSeek-R1-Distill 8B — 75

### Top by Language
1. Llama 3.2 1B — 100
2. Llama 3.2 3B — 100
3. Llama 3.1 8B — 100
4. Qwen2.5 0.5B — 100
5. Qwen2.5 3B — 100

### Top by Context
1. Llama 3.2 1B — 100
2. Llama 3.2 3B — 100
3. Llama 3.1 8B — 100
4. Gemma 3 4B — 100
5. Gemma 3 12B — 100

### Top by License
1. Llama 3.2 1B — 100
2. Llama 3.2 3B — 100
3. Llama 3.1 8B — 100
4. Qwen2.5 0.5B — 100
5. Qwen2.5 3B — 100
