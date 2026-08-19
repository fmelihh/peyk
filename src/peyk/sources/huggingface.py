"""Hugging Face source — cross-checks GGUF sizes for seed variants.

Best-effort and conservative: for families with a known GGUF repo hint, it sums
the size of the GGUF file matching the variant's quant via the public HF API.
Unknown families are skipped (returns nothing for them). Any failure degrades to
an empty list so curated/ollama data still stands.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import httpx

from ..models import ModelVariant

HF_API = "https://huggingface.co/api/models"

# family -> GGUF repo hint. Extend as the catalog grows.
REPO_HINTS: Dict[str, str] = {
    "llama 3.1": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct-GGUF",
    "qwen2.5-coder": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    "gemma 2": "bartowski/gemma-2-9b-it-GGUF",
    "phi-4": "bartowski/phi-4-GGUF",
    "mistral": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
}


class HuggingFaceSource:
    name = "huggingface"

    def __init__(
        self,
        seed: List[ModelVariant],
        client: Optional[httpx.Client] = None,
        timeout: float = 6.0,
    ) -> None:
        self._seed = seed
        self._client = client
        self._timeout = timeout

    def _gguf_size_gb(
        self, client: httpx.Client, repo: str, quant: str
    ) -> Optional[float]:
        resp = client.get(f"{HF_API}/{repo}", timeout=self._timeout)
        if resp.status_code != 200:
            return None
        siblings = resp.json().get("siblings", [])
        quant_l = quant.lower()
        for sib in siblings:
            fname = sib.get("rfilename", "")
            if fname.lower().endswith(".gguf") and quant_l in fname.lower():
                size = sib.get("size")
                if size:
                    return round(size / 1e9, 2)
        return None

    def fetch(self) -> List[ModelVariant]:
        owns_client = self._client is None
        client = self._client or httpx.Client()
        out: List[ModelVariant] = []
        try:
            for v in self._seed:
                repo = REPO_HINTS.get(v.family.lower())
                if not repo:
                    continue
                try:
                    size = self._gguf_size_gb(client, repo, v.quant)
                except (httpx.HTTPError, ValueError, KeyError):
                    continue
                if size is None:
                    continue
                out.append(v.model_copy(update={"file_size_gb": size, "source": "huggingface"}))
        finally:
            if owns_client:
                client.close()
        return out
