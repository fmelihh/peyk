"""Optional config file for team / on-prem defaults.

Looked up in order (first hit wins):
  1. ./.peyk.json                      (project-local, e.g. a repo default)
  2. $XDG_CONFIG_HOME/peyk/config.json (or ~/.config/peyk/config.json)

Recognised keys: languages, use_case, context, top, catalog_url. CLI flags
always override the config; the config overrides the built-in defaults. JSON is
used (not TOML) to avoid a dependency on Python 3.10.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_KEYS = {"languages", "use_case", "context", "top", "catalog_url"}


def _paths() -> list[Path]:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return [Path(".peyk.json"), Path(base) / "peyk" / "config.json"]


def load_config() -> dict:
    for p in _paths():
        try:
            if p.is_file():
                raw = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return {k: v for k, v in raw.items() if k in CONFIG_KEYS}
        except (OSError, ValueError):
            continue
    return {}
