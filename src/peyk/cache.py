"""Tiny TTL file cache for live-source results under ~/.cache/peyk.

Best-effort: any filesystem error silently disables the cache for that call, so
caching never breaks a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, List, Optional

from .models import ModelVariant

SIZES_TTL = 6 * 3600   # model sizes / availability
DISCOVER_TTL = 6 * 3600


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    d = Path(base) / "peyk"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(key: str) -> Path:
    return cache_dir() / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".json")


def read_fresh(key: str, ttl: float) -> Optional[list]:
    p = _path(key)
    try:
        if time.time() - p.stat().st_mtime > ttl:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write(key: str, data: list) -> None:
    try:
        _path(key).write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def key_for(name: str, seed_ids: List[str]) -> str:
    digest = hashlib.sha1("|".join(sorted(seed_ids)).encode("utf-8")).hexdigest()[:12]
    return f"{name}:{digest}"


def cached_variants(
    key: str,
    ttl: float,
    producer: Callable[[], List[ModelVariant]],
    use_cache: bool = True,
) -> List[ModelVariant]:
    """Return cached variants if fresh, else run `producer` and cache its output."""
    if use_cache:
        data = read_fresh(key, ttl)
        if data is not None:
            return [ModelVariant(**d) for d in data]
    out = producer()
    if use_cache and out:
        write(key, [v.model_dump(mode="json") for v in out])
    return out
