"""Command-line entry point.

Subcommands: recommend (default), hardware, plan, snippet. A bare `peyk` call
runs `recommend`, so existing usage is unchanged.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from rich.console import Console

from . import __version__, report
from .engine import recommend
from .profiler import detect
from .scoring import best_runnable_variant, weights_for
from .simulate import simulate_profile
from .sources import build_catalog

USE_CASES = ["chat", "coding", "summarize", "embedding"]
SUBCOMMANDS = {"recommend", "hardware", "plan", "snippet"}
SPEED_FLOOR = {"usable": 4.0, "fast": 10.0}


def _parse_size_gb(text: str | None) -> float | None:
    """Parse '1.5GB' / '512MB' / '8' (GB default) into gigabytes."""
    if not text:
        return None
    import re
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(gb|g|mb|m)?\s*$", text, re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid size '{text}' (use e.g. 1.5GB or 512MB)")
    val = float(m.group(1))
    unit = (m.group(2) or "gb").lower()
    return val / 1024 if unit in ("mb", "m") else val


def _add_hardware_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu", metavar="NAME",
                   help='Simulate a GPU, e.g. "RTX 4090", "2x RTX 5090", "A100 80GB"')
    p.add_argument("--gpu-only", action="store_true",
                   help="Use full VRAM with no OS reserve (with --gpu)")
    p.add_argument("--deep", action="store_true",
                   help="Run native probe scripts for richer hardware info")
    p.add_argument("--sudo", action="store_true",
                   help="Allow the deep probe to use sudo (Linux dmidecode); implies --deep")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="peyk",
        description="Recommend the best current local LLM models your hardware can run.",
    )
    p.add_argument("--version", action="version", version=f"peyk {__version__}")
    sub = p.add_subparsers(dest="command")

    rec = sub.add_parser("recommend", help="Recommend models for this (or a simulated) machine")
    rec.add_argument("--use-case", choices=USE_CASES, help="Intended use case (scoring weights)")
    rec.add_argument("--languages", default="en", help="Comma-separated language codes, e.g. tr,en")
    rec.add_argument("--context", type=int, default=8192, help="Target context length in tokens")
    rec.add_argument("--top", type=int, default=5, help="Models to show per criterion")
    rec.add_argument("--offline", action="store_true", help="Bundled catalog only; no network")
    rec.add_argument("--cross-check", action="store_true",
                     help="Verify sizes live via Ollama/HF (requires internet)")
    rec.add_argument("--discover", action="store_true",
                     help="Discover trending GGUF models from HuggingFace (requires internet)")
    rec.add_argument("--speed", choices=["usable", "fast"],
                     help="Hide models below a throughput floor (usable≥4, fast≥10 tok/s)")
    rec.add_argument("--vram-headroom", metavar="SIZE",
                     help="Extra memory to keep free, e.g. 1.5GB")
    rec.add_argument("--ram-budget", metavar="SIZE",
                     help="Cap the usable memory pool, e.g. 24GB")
    rec.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    rec.add_argument("--markdown", metavar="FILE", help="Write the report to a Markdown file")
    _add_hardware_flags(rec)

    hw = sub.add_parser("hardware", help="Show the detected (or simulated) hardware profile")
    hw.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    _add_hardware_flags(hw)

    pl = sub.add_parser("plan", help="Reverse lookup: what hardware runs a given model")
    pl.add_argument("model", help='Model to plan for, e.g. "llama 3.3 70b"')
    pl.add_argument("--context", type=int, default=8192, help="Target context length in tokens")
    pl.add_argument("--quant", default="Q4_K_M", help="Preferred quantization")
    pl.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    sn = sub.add_parser("snippet", help="Print ready-to-run commands for a model (no download)")
    sn.add_argument("model", help='Model to run, e.g. "qwen2.5 7b"')
    sn.add_argument("--context", type=int, default=8192, help="Target context length in tokens")
    sn.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    _add_hardware_flags(sn)

    return p


def _resolve_hardware(args, status: Console):
    """Detect the real machine or build a simulated one from --gpu."""
    if getattr(args, "gpu", None):
        try:
            return simulate_profile(args.gpu, gpu_only=getattr(args, "gpu_only", False))
        except ValueError as exc:
            status.print(f"[red]{exc}[/red]")
            raise SystemExit(2)
    deep = getattr(args, "deep", False) or getattr(args, "sudo", False)
    if deep:
        status.print("[dim]Running native hardware probe...[/dim]")
    return detect(deep=deep, allow_sudo=getattr(args, "sudo", False))


def _catalog_for(args, status: Console):
    offline = getattr(args, "offline", False) or not (
        getattr(args, "cross_check", False) or getattr(args, "discover", False)
    )
    if not offline:
        status.print("[dim]Fetching live data from sources (this may take a moment)...[/dim]")
    return build_catalog(
        offline=offline,
        cross_check=getattr(args, "cross_check", False),
        discover=getattr(args, "discover", False),
    )


def _cmd_recommend(args, console: Console, status: Console) -> int:
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    candidates = _catalog_for(args, status)
    if args.discover:
        discovered = sum(1 for c in candidates for v in c.variants
                         if v.source == "hf-discovered")
        if discovered:
            status.print(f"[dim]HuggingFace discovery: added {discovered} new variants.[/dim]")
    hw = _resolve_hardware(args, status)
    try:
        reserve = _parse_size_gb(args.vram_headroom)
        cap = _parse_size_gb(args.ram_budget)
    except ValueError as exc:
        status.print(f"[red]{exc}[/red]")
        return 2
    if reserve or cap is not None:
        hw = hw.model_copy(update={"reserve_gb": reserve or 0.0, "pool_cap_gb": cap})
    min_tps = SPEED_FLOOR.get(args.speed, 0.0)
    rec = recommend(hw=hw, candidates=candidates, context=args.context,
                    languages=languages, use_case=args.use_case, min_tps=min_tps)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(report.to_markdown(rec, top=args.top))
        status.print(f"[green]Markdown report written to:[/green] {args.markdown}")
    if args.json:
        print(report.to_json(rec))
    else:
        report.render_terminal(rec, top=args.top, console=console)
    return 0


def _cmd_hardware(args, console: Console, status: Console) -> int:
    hw = _resolve_hardware(args, status)
    if args.json:
        import json
        print(json.dumps(hw.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        report.render_hardware(hw, console=console)
    return 0


def _cmd_plan(args, console: Console, status: Console) -> int:
    from .planner import plan
    from .resolve import resolve_variant
    candidates = build_catalog(offline=True)
    variant = resolve_variant(args.model, candidates, prefer_quant=args.quant)
    if variant is None:
        status.print(f"[red]No catalog model matched '{args.model}'.[/red]")
        return 1
    result = plan(variant, context=args.context)
    if args.json:
        print(report.plan_to_json(result))
    else:
        report.render_plan(result, console=console)
    return 0


def _cmd_snippet(args, console: Console, status: Console) -> int:
    from .resolve import resolve_variant
    from .snippet import build_snippets
    candidates = build_catalog(offline=True)
    variant = resolve_variant(args.model, candidates)
    if variant is None:
        status.print(f"[red]No catalog model matched '{args.model}'.[/red]")
        return 1
    hw = _resolve_hardware(args, status)
    best = best_runnable_variant(
        [v for c in candidates for v in c.variants if c.family == variant.family
         and v.params_b == variant.params_b],
        hw, args.context, ["en"], weights_for(None),
    )
    chosen = best.variant if best else variant
    data = build_snippets(chosen, hw)
    if args.json:
        print(report.snippet_to_json(data))
    else:
        report.render_snippet(data, console=console)
    return 0


_DISPATCH = {
    "recommend": _cmd_recommend,
    "hardware": _cmd_hardware,
    "plan": _cmd_plan,
    "snippet": _cmd_snippet,
}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Default to `recommend` when no subcommand (and not a top-level flag) is given.
    if not argv or (argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help", "--version")):
        argv = ["recommend"] + argv

    args = _build_parser().parse_args(argv)
    console = Console()
    status = Console(stderr=True)
    return _DISPATCH[args.command](args, console, status)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
