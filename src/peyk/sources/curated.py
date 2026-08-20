"""Curated catalog — the offline source of truth for model metadata.

Loads the bundled JSON by default. To keep the catalog fresh without a code
release, it can instead pull a versioned JSON from a remote endpoint
(`PEYK_CATALOG_URL` or `--catalog-url`), cached locally, falling back to the
bundled copy on any failure. Either way it exposes the catalog's `updated` date
so the report can surface how stale it is.
"""

from __future__ import annotations

import json
import os
from importlib import resources

import httpx

from ..models import ModelVariant

ENV_CATALOG_URL = "PEYK_CATALOG_URL"


class CuratedSource:
    name = "curated"

    def __init__(self, path: str | None = None, url: str | None = None,
                 use_cache: bool = True) -> None:
        self._path = path
        self._url = url or os.environ.get(ENV_CATALOG_URL) or None
        self._use_cache = use_cache
        self.meta: dict = {}

    def _load_bundled(self) -> dict:
        if self._path:
            with open(self._path) as fh:
                return json.load(fh)
        data = resources.files("peyk.sources.data").joinpath("catalog.json")
        return json.loads(data.read_text(encoding="utf-8"))

    def _load_remote(self) -> dict | None:
        from .. import cache
        assert self._url is not None
        key = f"catalog:{self._url}"
        if self._use_cache:
            cached = cache.read_fresh(key, ttl=cache.SIZES_TTL)
            if isinstance(cached, dict):
                return cached
        try:
            resp = httpx.get(self._url, timeout=8.0)
            if resp.status_code != 200:
                return None
            raw = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        if "variants" not in raw:
            return None
        if self._use_cache:
            cache.write(key, raw)
        return raw

    def _load_raw(self) -> dict:
        if self._url:
            remote = self._load_remote()
            if remote is not None:
                remote.setdefault("origin", "remote")
                return remote
        raw = self._load_bundled()
        raw.setdefault("origin", "bundled")
        return raw

    def fetch(self) -> list[ModelVariant]:
        try:
            raw = self._load_raw()
        except (OSError, json.JSONDecodeError):
            return []
        self.meta = {
            "updated": raw.get("updated"),
            "version": raw.get("version"),
            "origin": raw.get("origin", "bundled"),
        }
        variants: list[ModelVariant] = []
        for item in raw.get("variants", []):
            try:
                variants.append(ModelVariant(source="curated", **item))
            except Exception:
                # Skip malformed rows rather than failing the whole catalog.
                continue
        return variants
