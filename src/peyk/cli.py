"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from rich.console import Console

from . import __version__
from .engine import recommend
from .profiler import detect
from .report import render_terminal, to_json, to_markdown
from .sources import build_catalog

USE_CASES = ["chat", "coding", "summarize", "embedding"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="peyk",
        description="Recommend the best current local LLM models your hardware can run.",
    )
    p.add_argument("--use-case", choices=USE_CASES,
                   help="Intended use case (adjusts scoring weights)")
    p.add_argument("--languages", default="en",
                   help="Comma-separated language codes, e.g. tr,en")
    p.add_argument("--context", type=int, default=8192,
                   help="Target context length in tokens (default: 8192)")
    p.add_argument("--top", type=int, default=5,
                   help="Number of models to show per criterion (default: 5)")
    p.add_argument("--offline", action="store_true",
                   help="Use the bundled catalog only; no network calls (default)")
    p.add_argument("--cross-check", action="store_true",
                   help="Verify sizes live via Ollama/HF (requires internet)")
    p.add_argument("--discover", action="store_true",
                   help="Discover trending GGUF models from HuggingFace (requires internet)")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    p.add_argument("--markdown", metavar="FILE", help="Write the report to a Markdown file")
    p.add_argument("--version", action="version", version=f"peyk {__version__}")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    console = Console()
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    wants_network = (args.cross_check or args.discover) and not args.offline
    offline = not wants_network
    if wants_network:
        console.print("[dim]Fetching live data from sources (this may take a moment)...[/dim]")
    candidates = build_catalog(
        offline=offline, cross_check=args.cross_check, discover=args.discover
    )
    if args.discover and not offline:
        discovered = sum(
            1 for c in candidates for v in c.variants if v.source == "hf-discovered"
        )
        console.print(f"[dim]HuggingFace discovery: added {discovered} new variants.[/dim]")

    hw = detect()
    rec = recommend(
        hw=hw,
        candidates=candidates,
        context=args.context,
        languages=languages,
        use_case=args.use_case,
    )

    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(to_markdown(rec, top=args.top))
        console.print(f"[green]Markdown report written to:[/green] {args.markdown}")

    if args.json:
        print(to_json(rec))
    else:
        render_terminal(rec, top=args.top, console=console)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
