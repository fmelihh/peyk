"""Core data models shared across the package."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# Fraction of total memory assumed usable for a model (rest = OS + other apps).
RAM_USABLE_FRACTION = 0.8
VRAM_USABLE_FRACTION = 0.9


class Accelerator(str, Enum):
    NONE = "NONE"
    NVIDIA = "NVIDIA"
    APPLE = "APPLE"
    AMD = "AMD"


class HardwareProfile(BaseModel):
    """Normalized description of the host machine."""

    os: str
    arch: str
    cpu_model: Optional[str] = None
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_flags: List[str] = Field(default_factory=list)
    ram_total_gb: float
    ram_available_gb: float
    ram_type: Optional[str] = None          # e.g. DDR5, LPDDR5 (deep probe)
    ram_speed_mtps: Optional[int] = None     # transfer rate in MT/s (deep probe)
    ram_channels: Optional[int] = None       # populated DIMMs / channels (deep probe)
    accelerator: Accelerator = Accelerator.NONE
    accelerator_name: Optional[str] = None
    gpu_count: int = 0
    vram_total_gb: float = 0.0  # aggregate across all GPUs of this accelerator
    unified_memory: bool = False
    mem_bandwidth_gbs: float = 0.0
    mem_bandwidth_source: str = "estimated"  # "measured" once derived from DIMM specs

    @property
    def memory_pool_gb(self) -> float:
        """Memory usable to hold model weights + KV cache.

        This answers "what can this machine run", so it is based on *capacity*
        (total RAM / VRAM minus an OS/runtime reserve) rather than the transient
        free amount — while never claiming less than what is free right now.
        Apple unified memory and CPU-only machines draw from RAM; a discrete
        accelerator draws from its own VRAM.
        """
        if self.unified_memory or self.accelerator == Accelerator.NONE:
            usable = self.ram_total_gb * RAM_USABLE_FRACTION
            return round(max(usable, self.ram_available_gb), 1)
        if self.vram_total_gb > 0:
            return round(self.vram_total_gb * VRAM_USABLE_FRACTION, 1)
        usable = self.ram_total_gb * RAM_USABLE_FRACTION
        return round(max(usable, self.ram_available_gb), 1)


class FitTier(str, Enum):
    FITS = "FITS"
    TIGHT = "TIGHT"
    NO_FIT = "NO_FIT"


class ModelVariant(BaseModel):
    """A single quantization of a model — the unit that fit/scoring runs on."""

    model_id: str
    family: str
    params_b: float
    quant: str = "Q4_K_M"
    file_size_gb: float
    context_max: int = 8192
    languages: List[str] = Field(default_factory=lambda: ["en"])
    license: str = "unknown"
    quality_score: float = 50.0  # 0-100 proxy for capability
    source: str = "curated"
    # Optional architecture hints for a sharper KV-cache estimate.
    n_layers: Optional[int] = None
    hidden: Optional[int] = None
    gqa_factor: float = 0.25  # modern GQA models cache far less than MHA (1.0)

    def merge_key(self) -> str:
        return f"{self.family.lower()}|{self.params_b}|{self.quant.lower()}"


class ModelCandidate(BaseModel):
    """A model family+size, holding its available quantization variants."""

    family: str
    params_b: float
    variants: List[ModelVariant] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.family.lower()}|{self.params_b}"


class FitResult(BaseModel):
    variant: ModelVariant
    mem_need_gb: float
    tier: FitTier
    est_tokens_per_sec: float


class ScoredModel(BaseModel):
    """A runnable variant with its fit result and per-criterion scores."""

    variant: ModelVariant
    fit: FitResult
    scores: dict  # criterion -> 0-100
    overall: float
