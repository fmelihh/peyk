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
        prog="inference-profiler",
        description="Donanımınıza en uygun güncel yerel LLM modellerini önerir.",
    )
    p.add_argument("--use-case", choices=USE_CASES, help="Kullanım amacı (skorlama ağırlıkları)")
    p.add_argument("--languages", default="en",
                   help="Virgülle ayrılmış dil kodları, ör: tr,en")
    p.add_argument("--context", type=int, default=8192,
                   help="Hedeflenen bağlam uzunluğu (token)")
    p.add_argument("--top", type=int, default=5, help="Kriter başına gösterilecek model sayısı")
    p.add_argument("--offline", action="store_true",
                   help="Sadece yerleşik katalog; ağ sorgusu yok (varsayılan)")
    p.add_argument("--cross-check", action="store_true",
                   help="Ollama/HF'den canlı boyut doğrulaması yap (internet gerekir)")
    p.add_argument("--discover", action="store_true",
                   help="HuggingFace'den trending GGUF modellerini keşfet (internet gerekir)")
    p.add_argument("--json", action="store_true", help="JSON çıktısı ver")
    p.add_argument("--markdown", metavar="FILE", help="Raporu Markdown dosyasına yaz")
    p.add_argument("--version", action="version", version=f"inference-profiler {__version__}")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    console = Console()
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    wants_network = (args.cross_check or args.discover) and not args.offline
    offline = not wants_network
    if wants_network:
        console.print("[dim]Canlı kaynaklardan veri alınıyor (bu biraz sürebilir)...[/dim]")
    candidates = build_catalog(
        offline=offline, cross_check=args.cross_check, discover=args.discover
    )
    if args.discover and not offline:
        discovered = sum(
            1 for c in candidates for v in c.variants if v.source == "hf-discovered"
        )
        console.print(f"[dim]HuggingFace keşfi: {discovered} yeni varyant eklendi.[/dim]")

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
        console.print(f"[green]Markdown raporu yazıldı:[/green] {args.markdown}")

    if args.json:
        print(to_json(rec))
    else:
        render_terminal(rec, top=args.top, console=console)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
