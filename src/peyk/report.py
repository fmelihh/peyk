"""Render recommendations to the terminal (rich), JSON, or Markdown."""

from __future__ import annotations

import json

from rich.box import ROUNDED, SIMPLE_HEAD
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import theme
from .engine import CRITERIA, Recommendation
from .models import FitTier, ScoredModel

TIER_STYLE = {FitTier.FITS: theme.GREEN, FitTier.TIGHT: theme.YELLOW, FitTier.NO_FIT: theme.RED}
SOURCE_LABEL = {
    "curated": "curated",
    "ollama": "ollama",
    "huggingface": "HF✓",
    "hf-discovered": "HF discover",
}
TIER_LABEL = {FitTier.FITS: "RUNS WELL", FitTier.TIGHT: "TIGHT", FitTier.NO_FIT: "WON'T FIT"}
CRITERION_LABEL = {
    "speed": "Speed",
    "quality": "Quality",
    "language": "Language",
    "context": "Context",
    "license": "License",
}


def _hardware_panel(hw) -> Panel:
    accel = hw.accelerator.value + (f" ({hw.accelerator_name})" if hw.accelerator_name else "")
    cpu_desc = f" {hw.cpu_model}" if hw.cpu_model else ""
    lines = [
        f"[bold]OS/Arch:[/bold] {hw.os} / {hw.arch}",
        f"[bold]CPU:[/bold]{cpu_desc} — {hw.cpu_cores_physical} cores "
        f"({hw.cpu_cores_logical} threads)  flags: {', '.join(hw.cpu_flags) or '-'}",
        f"[bold]RAM:[/bold] {hw.ram_available_gb:.1f} / {hw.ram_total_gb:.1f} GB free",
    ]
    if hw.disk_free_gb:
        lines.append(f"[bold]Disk free:[/bold] {hw.disk_free_gb:.1f} GB (for downloads)")
    if hw.ram_type or hw.ram_speed_mtps:
        ram_spec = " ".join(
            p for p in (hw.ram_type, f"{hw.ram_speed_mtps} MT/s" if hw.ram_speed_mtps else None,
                        f"x{hw.ram_channels} DIMM" if hw.ram_channels else None) if p
        )
        lines.append(f"[bold]RAM spec:[/bold] {ram_spec}")
    lines.append(f"[bold]Accelerator:[/bold] {accel}")
    if hw.vram_total_gb:
        gpu_note = f" (across {hw.gpu_count} GPUs)" if hw.gpu_count > 1 else ""
        lines.append(f"[bold]VRAM:[/bold] {hw.vram_total_gb:.1f} GB{gpu_note}")
    if hw.unified_memory:
        lines.append("[bold]Memory:[/bold] unified (shared RAM/VRAM pool)")
    bw_tag = {"measured": "measured", "simulated": "sim"}.get(hw.mem_bandwidth_source, "est.")
    lines.append(
        f"[bold]Usable memory pool (for sizing):[/bold] {hw.memory_pool_gb:.1f} GB  "
        f"| [bold]Bandwidth ({bw_tag}):[/bold] ~{hw.mem_bandwidth_gbs:.0f} GB/s"
    )
    title = ("🖥  Hardware [magenta]· SIMULATED[/magenta]" if hw.simulated
             else "🖥  Hardware")
    return Panel("\n".join(lines), title=title, title_align="left", box=ROUNDED,
                 border_style="magenta" if hw.simulated else theme.BORDER,
                 padding=(0, 2))


def render_hardware(hw, console: Console | None = None) -> None:
    (console or Console()).print(_hardware_panel(hw))


def _tier_label(tier: FitTier) -> Text:
    return theme.dot(TIER_STYLE[tier]) + Text(" " + TIER_LABEL[tier], style=TIER_STYLE[tier])


def _tier_table(rec: Recommendation) -> Table:
    table = Table(box=ROUNDED, border_style=theme.MUTED, header_style=f"bold {theme.ACCENT}",
                  title="Feasibility · best runnable quantization per model",
                  title_style=f"bold {theme.CYAN}", title_justify="left", expand=True,
                  row_styles=["", "on grey11"])
    table.add_column("Status", no_wrap=True)
    table.add_column("Model", style="bold", overflow="fold", min_width=12)
    table.add_column("Params", justify="right")
    table.add_column("Quant")
    table.add_column("Memory", justify="right")
    table.add_column("Speed t/s", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Source", no_wrap=True)

    for tier in (FitTier.FITS, FitTier.TIGHT, FitTier.NO_FIT):
        for s in rec.by_tier(tier):
            v = s.variant
            src = SOURCE_LABEL.get(v.source, v.source)
            src_text = Text(src, style="magenta" if v.source == "hf-discovered" else "dim")
            table.add_row(
                _tier_label(tier),
                v.family,
                f"{v.params_b:g}B",
                v.quant,
                f"{s.fit.mem_need_gb:.1f} GB",
                f"~{s.fit.est_tokens_per_sec:.0f}",
                Text(f"{s.overall:.0f}", style=f"bold {theme.ACCENT}"),
                src_text,
            )
    return table


_EVIDENCE_STYLE = {"live": "bright_green", "direct": "green",
                   "interpolated": "yellow", "family": "yellow", "proxy": "dim"}


def _criterion_table(rec: Recommendation, criterion: str, n: int) -> Table:
    icon = {"speed": "⚡", "quality": "★", "language": "🌐",
            "context": "▤", "license": "⚖"}.get(criterion, "")
    table = Table(box=SIMPLE_HEAD, header_style=f"bold {theme.ACCENT}", padding=(0, 1),
                  title=f"{icon} {CRITERION_LABEL.get(criterion, criterion)}",
                  title_style=f"bold {theme.CYAN}")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Model", style="bold")
    table.add_column("B", justify="right")
    table.add_column("Score", justify="right")
    show_conf = criterion == "quality"
    if show_conf:
        table.add_column("Ev.", no_wrap=True)
    for i, s in enumerate(rec.top_by(criterion, n=n), 1):
        row: list = [str(i), s.variant.family, f"{s.variant.params_b:g}",
                     Text(f"{s.scores.get(criterion, 0):.0f}", style=f"bold {theme.ACCENT}")]
        if show_conf:
            row.append(Text(s.quality_evidence,
                            style=_EVIDENCE_STYLE.get(s.quality_evidence, "dim")))
        table.add_row(*row)
    return table


def _disk_warning(rec: Recommendation) -> str | None:
    """Flag models that fit in memory but exceed free disk (must download first)."""
    disk = rec.hw.disk_free_gb
    if disk <= 0:
        return None
    runnable = [s for s in rec.scored if s.fit.tier != FitTier.NO_FIT]
    too_big = [s for s in runnable if s.variant.file_size_gb > disk]
    if not too_big:
        return None
    biggest = max(too_big, key=lambda s: s.variant.file_size_gb)
    return (f"⚠ {len(too_big)} runnable model(s) exceed free disk ({disk:.1f} GB) — "
            f"e.g. {biggest.variant.family} needs {biggest.variant.file_size_gb:.1f} GB to download.")


def render_terminal(rec: Recommendation, top: int = 5, console: Console | None = None) -> None:
    console = console or Console()
    console.print()
    console.print(theme.header())
    console.print()
    console.print(_hardware_panel(rec.hw))
    console.print()
    console.print(_tier_table(rec))
    disk_warn = _disk_warning(rec)
    if disk_warn:
        console.print(Text(disk_warn, style=theme.YELLOW))
    console.print()
    console.rule(Text("Top models by criterion", style=f"bold {theme.CYAN}"),
                 style=theme.MUTED)
    console.print()
    console.print(Columns([_criterion_table(rec, c, top) for c in CRITERIA],
                          equal=False, expand=True, padding=(0, 2)))
    console.print(
        "\n[dim]Estimates: memory & speed are approximate and depend on backend, "
        "quantization, and context. Quality is an evidence-tagged benchmark score "
        "(live › direct › interpolated › proxy), discounted by confidence.[/dim]"
    )


def _scored_to_dict(s: ScoredModel) -> dict:
    return {
        "model": s.variant.family,
        "model_id": s.variant.model_id,
        "params_b": s.variant.params_b,
        "quant": s.variant.quant,
        "source": s.variant.source,
        "tier": s.fit.tier.value,
        "mem_need_gb": s.fit.mem_need_gb,
        "est_tokens_per_sec": s.fit.est_tokens_per_sec,
        "scores": s.scores,
        "quality_evidence": s.quality_evidence,
        "quality_source": s.quality_source,
        "overall": s.overall,
    }


def to_json(rec: Recommendation) -> str:
    payload = {
        "hardware": rec.hw.model_dump(mode="json"),
        "context": rec.context,
        "models": [_scored_to_dict(s) for s in
                   sorted(rec.scored, key=lambda x: x.overall, reverse=True)],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _gpu_label(spec) -> str:
    return f"{spec.name.upper()} ({spec.vram_gb:g} GB)" if spec else "—"


def render_plan(result, console: Console | None = None) -> None:
    console = console or Console()
    v = result.variant
    lines = [
        f"[bold]Model:[/bold] {v.family} {v.params_b:g}B {v.quant}  "
        f"(context {result.context})",
        f"[bold]Memory needed:[/bold] {result.mem_need_gb:.1f} GB "
        f"(weights + KV cache + overhead)",
        "",
        f"[bold]To fit comfortably:[/bold] ≥ {result.min_vram_fits_gb:.1f} GB VRAM",
        f"[bold]To run (tight):[/bold] ≥ {result.min_vram_tight_gb:.1f} GB VRAM",
        f"[bold]CPU / unified RAM needed:[/bold] ~{result.ram_needed_gb:.1f} GB",
        "",
        f"[bold]Cheapest GPU that fits:[/bold] {_gpu_label(result.cheapest_fits)}",
        f"[bold]Cheapest GPU (tight):[/bold] {_gpu_label(result.cheapest_tight)}",
    ]
    if result.cheapest_fits is None and result.multi_gpu:
        mg = result.multi_gpu
        lines.append(
            f"[bold]Multi-GPU option:[/bold] {mg.count}x {mg.spec.name.upper()} "
            f"({mg.spec.vram_gb:g} GB each)"
        )
    console.print(Panel("\n".join(lines), title="🔎  peyk plan", title_align="left",
                        box=ROUNDED, border_style=theme.BORDER, padding=(0, 2)))


def plan_to_json(result) -> str:
    r = result
    payload = {
        "model": r.variant.family, "model_id": r.variant.model_id,
        "params_b": r.variant.params_b, "quant": r.variant.quant, "context": r.context,
        "mem_need_gb": r.mem_need_gb,
        "min_vram_fits_gb": r.min_vram_fits_gb, "min_vram_tight_gb": r.min_vram_tight_gb,
        "ram_needed_gb": r.ram_needed_gb,
        "cheapest_fits": r.cheapest_fits.name if r.cheapest_fits else None,
        "cheapest_tight": r.cheapest_tight.name if r.cheapest_tight else None,
        "multi_gpu": (f"{r.multi_gpu.count}x {r.multi_gpu.spec.name}"
                      if r.multi_gpu else None),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def render_snippet(data: dict, console: Console | None = None) -> None:
    console = console or Console()
    v = f"{data['model']} {data['params_b']:g}B {data['quant']}"
    console.print(Panel(f"[bold]{v}[/bold]  ([dim]{data['model_id']}[/dim])",
                        title="⚡  peyk snippet", title_align="left", box=ROUNDED,
                        border_style=theme.BORDER, padding=(0, 2)))
    for title, cmd in data["commands"].items():
        console.print(Panel(cmd, title=f"[bold {theme.CYAN}]{title}[/]", title_align="left",
                            box=ROUNDED, border_style=theme.MUTED, padding=(0, 1)))


def snippet_to_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def to_markdown(rec: Recommendation, top: int = 5) -> str:
    lines: list[str] = ["# peyk — LLM model recommendation", ""]
    hw = rec.hw
    lines += [
        "## Hardware",
        f"- OS/Arch: {hw.os} / {hw.arch}",
        f"- RAM: {hw.ram_available_gb:.1f} / {hw.ram_total_gb:.1f} GB",
        f"- Accelerator: {hw.accelerator.value} {hw.accelerator_name or ''}".rstrip(),
        f"- Usable memory pool: {hw.memory_pool_gb:.1f} GB (~{hw.mem_bandwidth_gbs:.0f} GB/s)",
        "",
        "## Feasibility",
        "",
        "| Status | Model | Params | Quant | Memory | Speed (est.) | Score | Source |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for tier in (FitTier.FITS, FitTier.TIGHT, FitTier.NO_FIT):
        for s in rec.by_tier(tier):
            v = s.variant
            src = SOURCE_LABEL.get(v.source, v.source)
            lines.append(
                f"| {TIER_LABEL[tier]} | {v.family} | {v.params_b:g}B | {v.quant} | "
                f"{s.fit.mem_need_gb:.1f} GB | ~{s.fit.est_tokens_per_sec:.0f} tok/s "
                f"| {s.overall:.0f} | {src} |"
            )
    lines.append("")
    for criterion in CRITERIA:
        lines.append(f"### Top by {CRITERION_LABEL.get(criterion, criterion)}")
        for i, s in enumerate(rec.top_by(criterion, n=top), 1):
            lines.append(f"{i}. {s.variant.family} {s.variant.params_b:g}B — "
                         f"{s.scores.get(criterion, 0):.0f}")
        lines.append("")
    return "\n".join(lines)
