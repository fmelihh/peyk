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
TIER_LABEL = {FitTier.FITS: "RAHAT ÇALIŞIR", FitTier.TIGHT: "ZORLAR", FitTier.NO_FIT: "SIĞMAZ"}
CRITERION_LABEL = {
    "speed": "Hız",
    "quality": "Kalite",
    "language": "Dil desteği",
    "context": "Bağlam",
    "license": "Lisans",
}


def _hardware_panel(rec: Recommendation) -> Panel:
    hw = rec.hw
    lines = [
        f"[bold]OS/Arch:[/bold] {hw.os} / {hw.arch}",
        f"[bold]CPU:[/bold] {hw.cpu_cores_physical} çekirdek "
        f"({hw.cpu_cores_logical} thread)  flags: {', '.join(hw.cpu_flags) or '-'}",
        f"[bold]RAM:[/bold] {hw.ram_available_gb:.1f} / {hw.ram_total_gb:.1f} GB uygun",
        f"[bold]Hızlandırıcı:[/bold] {hw.accelerator.value}"
        + (f" ({hw.accelerator_name})" if hw.accelerator_name else ""),
    ]
    if hw.vram_total_gb:
        lines.append(f"[bold]VRAM:[/bold] {hw.vram_total_gb:.1f} GB")
    if hw.unified_memory:
        lines.append("[bold]Bellek:[/bold] unified (RAM havuzu paylaşımlı)")
    lines.append(
        f"[bold]Bellek havuzu (öneri için):[/bold] {hw.memory_pool_gb:.1f} GB  "
        f"| [bold]Bant genişliği (tahmini):[/bold] ~{hw.mem_bandwidth_gbs:.0f} GB/s"
    )
    return Panel("\n".join(lines), title="Donanım Profili", border_style="cyan")


def _tier_table(rec: Recommendation) -> Table:
    table = Table(title="Uygunluk (en iyi çalışabilir quantization ile)")
    table.add_column("Durum", no_wrap=True)
    table.add_column("Model", style="bold")
    table.add_column("Params", justify="right")
    table.add_column("Quant")
    table.add_column("Bellek ihtiyacı", justify="right")
    table.add_column("Hız (tahmini)", justify="right")
    table.add_column("Genel", justify="right")

    for tier in (FitTier.FITS, FitTier.TIGHT, FitTier.NO_FIT):
        for s in rec.by_tier(tier):
            v = s.variant
            table.add_row(
                Text(TIER_LABEL[tier], style=TIER_STYLE[tier]),
                v.family,
                f"{v.params_b:g}B",
                v.quant,
                f"{s.fit.mem_need_gb:.1f} GB",
                f"~{s.fit.est_tokens_per_sec:.0f} tok/s",
                f"{s.overall:.0f}",
            )
    return table


def _criterion_table(rec: Recommendation, criterion: str, n: int) -> Table:
    table = Table(title=f"En iyi: {CRITERION_LABEL.get(criterion, criterion)}")
    table.add_column("#", justify="right")
    table.add_column("Model", style="bold")
    table.add_column("Params", justify="right")
    table.add_column("Puan", justify="right")
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
        "\n[dim]Not: bellek ve hız değerleri kaba tahmindir; gerçek sonuç "
        "backend, quantization ve bağlam uzunluğuna göre değişir.[/dim]"
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
    lines: List[str] = ["# LLM Model Önerisi", ""]
    hw = rec.hw
    lines += [
        "## Donanım",
        f"- OS/Arch: {hw.os} / {hw.arch}",
        f"- RAM: {hw.ram_available_gb:.1f} / {hw.ram_total_gb:.1f} GB",
        f"- Hızlandırıcı: {hw.accelerator.value} {hw.accelerator_name or ''}".rstrip(),
        f"- Bellek havuzu: {hw.memory_pool_gb:.1f} GB (~{hw.mem_bandwidth_gbs:.0f} GB/s)",
        "",
        "## Uygunluk",
        "",
        "| Durum | Model | Params | Quant | Bellek | Hız (tahmini) | Genel |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for tier in (FitTier.FITS, FitTier.TIGHT, FitTier.NO_FIT):
        for s in rec.by_tier(tier):
            v = s.variant
            lines.append(
                f"| {TIER_LABEL[tier]} | {v.family} | {v.params_b:g}B | {v.quant} | "
                f"{s.fit.mem_need_gb:.1f} GB | ~{s.fit.est_tokens_per_sec:.0f} tok/s | {s.overall:.0f} |"
            )
    lines.append("")
    for criterion in CRITERIA:
        lines.append(f"### En iyi: {CRITERION_LABEL.get(criterion, criterion)}")
        for i, s in enumerate(rec.top_by(criterion, n=top), 1):
            lines.append(f"{i}. {s.variant.family} {s.variant.params_b:g}B — "
                         f"{s.scores.get(criterion, 0):.0f}")
        lines.append("")
    return "\n".join(lines)
