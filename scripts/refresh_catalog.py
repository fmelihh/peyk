#!/usr/bin/env python3
"""Refresh the bundled catalog's file sizes from live registries and bump its date.

Run weekly (see .github/workflows/refresh-catalog.yml) so the catalog doesn't go
stale silently. It only touches `file_size_gb` (verified against the Ollama
registry) and the `updated` / `version` fields — curated metadata (quality,
languages, license) is never overwritten.

Usage:
    python scripts/refresh_catalog.py            # rewrite catalog.json
    python scripts/refresh_catalog.py --dry-run  # report changes, exit 1 if any
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "src" / "peyk" / "sources" / "data" / "catalog.json"
SIZE_EPS_GB = 0.05

sys.path.insert(0, str(REPO / "src"))

from peyk.models import ModelVariant  # noqa: E402
from peyk.sources.ollama import OllamaSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Report changes, don't write")
    args = ap.parse_args()

    raw = json.loads(CATALOG.read_text(encoding="utf-8"))
    variants = [ModelVariant(**{k: v for k, v in item.items()})
                for item in raw.get("variants", [])]

    live = OllamaSource(seed=variants, timeout=8.0).fetch()
    live_size = {v.model_id: v.file_size_gb for v in live}

    changes: list[str] = []
    for item in raw.get("variants", []):
        mid = item.get("model_id")
        if mid in live_size:
            new = live_size[mid]
            old = float(item.get("file_size_gb", 0))
            if abs(new - old) > SIZE_EPS_GB:
                changes.append(f"  {mid}: {old:.2f} -> {new:.2f} GB")
                item["file_size_gb"] = new

    print(f"Checked {len(variants)} models, {len(live_size)} confirmed by Ollama.")
    if changes:
        print(f"{len(changes)} size change(s):")
        print("\n".join(changes))
    else:
        print("No size changes.")

    if args.dry_run:
        return 1 if changes else 0

    # Always bump the freshness stamp on a real run.
    today = date.today()
    raw["updated"] = today.isoformat()
    raw["version"] = today.strftime("%Y.%m.%d")
    CATALOG.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {CATALOG.relative_to(REPO)} (updated {raw['updated']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
