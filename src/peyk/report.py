"""Render recommendations to the terminal (rich), JSON, or Markdown."""

from __future__ import annotations

import json
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .engine import CRITERIA, Recommendation
from .models import FitTier, ScoredModel

TIER_STYLE = {FitTier.FITS: "green", FitTier.TIGHT: "yellow", FitTier.NO_FIT: "red"}
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


def _hardware_panel(rec: Recommendation) -> Panel:
    hw = rec.hw
    accel = hw.accelerator.value + (f" ({hw.accelerator_name})" if hw.accelerator_name else "")
    cpu_desc = f" {hw.cpu_model}" if hw.cpu_model else ""
    lines = [
        f"[bold]OS/Arch:[/bold] {hw.os} / {hw.arch}",
        f"[bold]CPU:[/bold]{cpu_desc} — {hw.cpu_cores_physical} cores "
        f"({hw.cpu_cores_logical} threads)  flags: {', '.join(hw.cpu_flags) or '-'}",
        f"[bold]RAM:[/bold] {hw.ram_available_gb:.1f} / {hw.ram_total_gb:.1f} GB free",
    ]
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
    bw_tag = "measured" if hw.mem_bandwidth_source == "measured" else "est."
    lines.append(
        f"[bold]Usable memory pool (for sizing):[/bold] {hw.memory_pool_gb:.1f} GB  "
        f"| [bold]Bandwidth ({bw_tag}):[/bold] ~{hw.mem_bandwidth_gbs:.0f} GB/s"
    )
    return Panel("\n".join(lines), title="Hardware Profile", border_style="cyan")


def _tier_table(rec: Recommendation) -> Table:
    table = Table(title="Feasibility (using each model's best runnable quantization)")
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
                Text(TIER_LABEL[tier], style=TIER_STYLE[tier]),
                v.family,
                f"{v.params_b:g}B",
                v.quant,
                f"{s.fit.mem_need_gb:.1f} GB",
                f"~{s.fit.est_tokens_per_sec:.0f}",
                f"{s.overall:.0f}",
                src_text,
            )
    return table


def _criterion_table(rec: Recommendation, criterion: str, n: int) -> Table:
    table = Table(title=f"Top by {CRITERION_LABEL.get(criterion, criterion)}")
    table.add_column("#", justify="right")
    table.add_column("Model", style="bold")
    table.add_column("Params", justify="right")
    table.add_column("Score", justify="right")
    for i, s in enumerate(rec.top_by(criterion, n=n), 1):
        table.add_row(str(i), s.variant.family, f"{s.variant.params_b:g}B",
                      f"{s.scores.get(criterion, 0):.0f}")
    return table


def render_terminal(rec: Recommendation, top: int = 5, console: Console | None = None) -> None:
    console = console or Console()
    console.print(_hardware_panel(rec))
    console.print()
    console.print(_tier_table(rec))
    console.print()
    for criterion in CRITERIA:
        console.print(_criterion_table(rec, criterion, top))
    console.print(
        "\n[dim]Note: memory and speed figures are rough estimates; real results "
        "depend on the backend, quantization, and context length. For 'HF discover' "
        "models the quality score is estimated from parameter count only (no benchmark)."
        "[/dim]"
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


def to_markdown(rec: Recommendation, top: int = 5) -> str:
    lines: List[str] = ["# peyk — LLM model recommendation", ""]
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
