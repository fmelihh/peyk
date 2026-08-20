"""Core data models shared across the package."""

from __future__ import annotations

from enum import Enum

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
    cpu_model: str | None = None
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_flags: list[str] = Field(default_factory=list)
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float = 0.0                # free space where models download
    ram_type: str | None = None          # e.g. DDR5, LPDDR5 (deep probe)
    ram_speed_mtps: int | None = None     # transfer rate in MT/s (deep probe)
    ram_channels: int | None = None       # populated DIMMs / channels (deep probe)
    accelerator: Accelerator = Accelerator.NONE
    accelerator_name: str | None = None
    gpu_count: int = 0
    vram_total_gb: float = 0.0  # aggregate across all GPUs of this accelerator
    unified_memory: bool = False
    mem_bandwidth_gbs: float = 0.0
    mem_bandwidth_source: str = "estimated"  # "measured"/"simulated" when applicable
    vram_usable_fraction: float = VRAM_USABLE_FRACTION  # 1.0 under --gpu-only
    simulated: bool = False
    reserve_gb: float = 0.0            # extra headroom subtracted from the pool
    pool_cap_gb: float | None = None  # hard cap on the usable pool (budget)

    @property
    def memory_pool_gb(self) -> float:
        """Memory usable to hold model weights + KV cache.

        This answers "what can this machine run", so it is based on *capacity*
        (total RAM / VRAM minus an OS/runtime reserve) rather than the transient
        free amount — while never claiming less than what is free right now.
        Apple unified memory and CPU-only machines draw from RAM; a discrete
        accelerator draws from its own VRAM. User budgets (`pool_cap_gb`,
        `reserve_gb`) are applied last.
        """
        if self.unified_memory or self.accelerator == Accelerator.NONE:
            base = max(self.ram_total_gb * RAM_USABLE_FRACTION, self.ram_available_gb)
        elif self.vram_total_gb > 0:
            base = self.vram_total_gb * self.vram_usable_fraction
        else:
            base = max(self.ram_total_gb * RAM_USABLE_FRACTION, self.ram_available_gb)

        if self.pool_cap_gb is not None:
            base = min(base, self.pool_cap_gb)
        return round(max(0.0, base - self.reserve_gb), 1)


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
    active_params_b: float | None = None  # MoE: active (< total) params per token
    modality: str = "text"                   # "text" | "vision" (multimodal)
    context_max: int = 8192
    languages: list[str] = Field(default_factory=lambda: ["en"])
    license: str = "unknown"
    quality_score: float = 50.0  # 0-100 proxy for capability
    source: str = "curated"
    # Optional architecture hints for a sharper KV-cache estimate.
    n_layers: int | None = None
    hidden: int | None = None
    gqa_factor: float = 0.25  # modern GQA models cache far less than MHA (1.0)

    def merge_key(self) -> str:
        return f"{self.family.lower()}|{self.params_b}|{self.quant.lower()}"


class ModelCandidate(BaseModel):
    """A model family+size, holding its available quantization variants."""

    family: str
    params_b: float
    variants: list[ModelVariant] = Field(default_factory=list)

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
    quality_evidence: str = "proxy"  # direct | interpolated | family | proxy
    quality_source: str | None = None
