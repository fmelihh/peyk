"""Bundled curated catalog — the offline source of truth for metadata."""

from __future__ import annotations

import json
from importlib import resources
from typing import List

from ..models import ModelVariant


class CuratedSource:
    name = "curated"

    def __init__(self, path: str | None = None) -> None:
        self._path = path

    def _load_raw(self) -> dict:
        if self._path:
            with open(self._path, "r") as fh:
                return json.load(fh)
        data = resources.files("inference_profiler.sources.data").joinpath("catalog.json")
        return json.loads(data.read_text(encoding="utf-8"))

    def fetch(self) -> List[ModelVariant]:
        try:
            raw = self._load_raw()
        except (OSError, json.JSONDecodeError):
            return []
        variants: List[ModelVariant] = []
        for item in raw.get("variants", []):
            try:
                variants.append(ModelVariant(source="curated", **item))
            except Exception:
                # Skip malformed rows rather than failing the whole catalog.
                continue
        return variants
