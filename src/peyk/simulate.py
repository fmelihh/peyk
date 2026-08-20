"""Build a synthetic HardwareProfile for `--gpu` simulation.

Keeps the host's real CPU/RAM (they still matter for offload) but swaps in a
simulated discrete GPU so you can ask "what would run on an RTX 5090 before I buy
one".
"""

from __future__ import annotations

from .gpus import parse_gpu_arg
from .models import VRAM_USABLE_FRACTION, HardwareProfile
from .profiler import detect


def simulate_profile(
    gpu_arg: str, gpu_only: bool = False, base: HardwareProfile | None = None
) -> HardwareProfile:
    spec, count = parse_gpu_arg(gpu_arg)
    base = base if base is not None else detect()
    total_vram = round(spec.vram_gb * count, 1)
    label = spec.name.upper() if count == 1 else f"{count}x {spec.name.upper()}"
    return base.model_copy(update={
        "accelerator": spec.vendor,
        "accelerator_name": label,
        "gpu_count": count,
        "vram_total_gb": total_vram,
        "unified_memory": False,
        "mem_bandwidth_gbs": spec.bandwidth_gbs,
        "mem_bandwidth_source": "simulated",
        "vram_usable_fraction": 1.0 if gpu_only else VRAM_USABLE_FRACTION,
        "simulated": True,
    })
