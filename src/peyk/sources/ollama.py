"""Ollama registry source — cross-checks exact GGUF sizes for seed variants.

Best-effort: queries the public Ollama registry for each seed model's manifest
and returns copies with the real on-disk size. Any failure for a given model is
skipped; a total failure yields an empty list so the pipeline falls back to
curated data.
"""

from __future__ import annotations

from typing import List, Optional

import httpx

from ..models import ModelVariant

REGISTRY = "https://registry.ollama.ai/v2/library"
MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"


class OllamaSource:
    name = "ollama"

    def __init__(
        self,
        seed: List[ModelVariant],
        client: Optional[httpx.Client] = None,
        timeout: float = 6.0,
    ) -> None:
        self._seed = seed
        self._client = client
        self._timeout = timeout

    def _manifest_size_gb(self, client: httpx.Client, name: str, tag: str) -> Optional[float]:
        url = f"{REGISTRY}/{name}/manifests/{tag}"
        resp = client.get(url, headers={"Accept": MANIFEST_ACCEPT}, timeout=self._timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        for layer in data.get("layers", []):
            if layer.get("mediaType") == MODEL_MEDIA_TYPE:
                size = layer.get("size")
                if size:
                    return round(size / 1e9, 2)
        return None

    def fetch(self) -> List[ModelVariant]:
        owns_client = self._client is None
        client = self._client or httpx.Client()
        out: List[ModelVariant] = []
        try:
            for v in self._seed:
                if ":" not in v.model_id:
                    continue
                name, tag = v.model_id.split(":", 1)
                try:
                    size = self._manifest_size_gb(client, name, tag)
                except (httpx.HTTPError, ValueError, KeyError):
                    continue
                if size is None:
                    continue
                out.append(v.model_copy(update={"file_size_gb": size, "source": "ollama"}))
        finally:
            if owns_client:
                client.close()
        return out
