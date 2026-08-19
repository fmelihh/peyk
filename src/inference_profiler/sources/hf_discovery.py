"""Hugging Face discovery — auto-surface trending GGUF models we don't curate.

This is additive and opt-in. It lists popular GGUF repos, extracts one variant
per quantization (summing split files), and derives what metadata it can from
the repo id and tags. Families already covered by the curated catalog are
skipped, so discovery only brings *new* models to the table. Quality is a coarse
params-based estimate (there is no benchmark signal here) and every variant is
tagged `source="hf-discovered"` so it can be told apart from curated data.

Best-effort: any failure yields an empty list and the pipeline stands on
whatever else succeeded.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import httpx

from ..models import ModelVariant

HF_LIST = "https://huggingface.co/api/models"

_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
_QUANT_RE = re.compile(r"(IQ\d[\w]*|Q\d[\w]*|BF16|FP16|F16|F32)", re.IGNORECASE)
_KEYWORD_RE = re.compile(r"[a-z]+")

# GGUF exists for non-text-generation models too; keep those out of an LLM report.
_NON_LLM_TAGS = {
    "automatic-speech-recognition", "audio", "audio-to-audio", "text-to-speech",
    "image-text-to-text", "image-to-text", "sentence-similarity", "feature-extraction",
    "text-to-image", "text-ranking", "reranker",
}
_NON_LLM_NAME = re.compile(
    r"(asr|ctc|parakeet|whisper|tts|speech|embed|rerank|bge|clip|vit|vision|stable-?diffusion)",
    re.IGNORECASE,
)


def estimated_quality(params_b: float) -> float:
    """Coarse capability proxy from parameter count (no benchmark available)."""
    for threshold, score in (
        (1, 35), (3, 50), (5, 60), (8, 68), (15, 77), (34, 84), (70, 88),
    ):
        if params_b < threshold:
            return float(score)
    return 90.0


def parse_params_b(repo_id: str) -> Optional[float]:
    matches = _PARAMS_RE.findall(repo_id)
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def parse_quant(filename: str) -> Optional[str]:
    m = _QUANT_RE.search(filename)
    return m.group(1).upper() if m else None


def _family_from_repo(repo_id: str) -> str:
    name = repo_id.split("/")[-1]
    name = re.sub(r"[-_]?GGUF$", "", name, flags=re.IGNORECASE)
    return name


def _curated_keywords(exclude_families: Iterable[str]) -> set[str]:
    keywords = set()
    for fam in exclude_families:
        m = _KEYWORD_RE.search(fam.lower())
        if m:
            keywords.add(m.group(0))
    return keywords


class HuggingFaceDiscoverySource:
    name = "hf-discovered"

    def __init__(
        self,
        exclude_families: Optional[Iterable[str]] = None,
        client: Optional[httpx.Client] = None,
        limit: int = 15,
        timeout: float = 8.0,
    ) -> None:
        self._exclude = _curated_keywords(exclude_families or [])
        self._client = client
        self._limit = limit
        self._timeout = timeout

    def _list_repos(self, client: httpx.Client) -> List[dict]:
        params = {
            "filter": ["gguf", "text-generation"],
            "sort": "downloads",
            "direction": "-1",
            "limit": str(self._limit),
            "expand[]": ["tags", "downloads"],
        }
        resp = client.get(HF_LIST, params=params, timeout=self._timeout)
        if resp.status_code != 200:
            return []
        return resp.json()

    def _repo_files(self, client: httpx.Client, repo: str) -> List[dict]:
        url = f"{HF_LIST}/{repo}/tree/main"
        resp = client.get(url, params={"recursive": "true"}, timeout=self._timeout)
        if resp.status_code != 200:
            return []
        return resp.json()

    def _is_excluded(self, repo_id: str) -> bool:
        low = repo_id.lower()
        return any(kw in low for kw in self._exclude)

    @staticmethod
    def _tags_meta(tags: List[str]) -> tuple[str, List[str]]:
        license_ = "unknown"
        languages: List[str] = []
        for t in tags:
            if t.startswith("license:"):
                license_ = t.split(":", 1)[1]
            elif t.startswith("language:"):
                languages.append(t.split(":", 1)[1])
        return license_, (languages or ["en"])

    def _variants_for_repo(self, client: httpx.Client, entry: dict) -> List[ModelVariant]:
        repo_id = entry.get("id") or entry.get("modelId")
        if not repo_id or self._is_excluded(repo_id):
            return []
        tags = entry.get("tags", []) or []
        if _NON_LLM_NAME.search(repo_id) or (set(tags) & _NON_LLM_TAGS):
            return []
        params_b = parse_params_b(repo_id)
        if params_b is None:
            return []
        license_, languages = self._tags_meta(tags)
        family = _family_from_repo(repo_id)

        # Sum sizes per quant so split files (00001-of-00002) count once.
        sizes: Dict[str, float] = defaultdict(float)
        for f in self._repo_files(client, repo_id):
            path = f.get("path", "")
            if f.get("type") != "file" or not path.lower().endswith(".gguf"):
                continue
            quant = parse_quant(path)
            if quant is None:
                continue
            sizes[quant] += (f.get("size") or 0) / 1e9

        out: List[ModelVariant] = []
        for quant, size_gb in sizes.items():
            if size_gb <= 0:
                continue
            out.append(ModelVariant(
                model_id=f"hf:{repo_id}:{quant}",
                family=family,
                params_b=params_b,
                quant=quant,
                file_size_gb=round(size_gb, 2),
                languages=languages,
                license=license_,
                quality_score=estimated_quality(params_b),
                source="hf-discovered",
            ))
        return out

    def fetch(self) -> List[ModelVariant]:
        owns_client = self._client is None
        client = self._client or httpx.Client()
        out: List[ModelVariant] = []
        try:
            try:
                repos = self._list_repos(client)
            except (httpx.HTTPError, ValueError):
                return []
            for entry in repos:
                try:
                    out.extend(self._variants_for_repo(client, entry))
                except (httpx.HTTPError, ValueError, KeyError):
                    continue
        finally:
            if owns_client:
                client.close()
        return out
