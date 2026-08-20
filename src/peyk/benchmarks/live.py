"""Optional live benchmark tier.

Frozen snapshots go stale as new model generations ship. When enabled, peyk
fetches a fresh benchmark table from a configurable JSON endpoint and overlays it
on the frozen tier with higher trust. Off by default; best-effort (any failure
falls back to frozen data).

Endpoint contract: JSON `{"entries": [{"family": str, "params_b": num,
"quality": num, "elo"?: num, "source"?: str}, ...]}` — the same shape as the
bundled frozen snapshot.
"""

from __future__ import annotations

import os
from typing import List, Optional

import httpx

ENV_URL = "PEYK_BENCHMARKS_URL"


def resolve_url(url: Optional[str] = None) -> Optional[str]:
    return url or os.environ.get(ENV_URL) or None


def fetch_live(
    url: Optional[str] = None,
    client: Optional[httpx.Client] = None,
    timeout: float = 8.0,
) -> List[dict]:
    """Fetch live benchmark entries. Returns [] when disabled or on any error."""
    target = resolve_url(url)
    if not target:
        return []
    owns = client is None
    client = client or httpx.Client()
    try:
        resp = client.get(target, timeout=timeout)
        if resp.status_code != 200:
            return []
        entries = resp.json().get("entries", [])
        return [e for e in entries if "family" in e and "params_b" in e and "quality" in e]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return []
    finally:
        if owns:
            client.close()
