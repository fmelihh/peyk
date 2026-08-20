"""Shared visual theme: the peyk palette, gradient text, and header.

Everything here is pure rich — no extra dependencies. The palette echoes the
logo (indigo -> cyan) for a soft, consistent look.
"""

from __future__ import annotations

from rich.align import Align
from rich.box import ROUNDED
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

# Palette (logo gradient + neutrals)
INDIGO = "#6366f1"
BLUE = "#4f8ff7"
CYAN = "#22d3ee"
MUTED = "grey62"
GREEN = "#34d399"
YELLOW = "#fbbf24"
RED = "#f87171"

ACCENT = INDIGO
BORDER = INDIGO

GRADIENT = [INDIGO, BLUE, CYAN]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def gradient_text(text: str, stops: list[str] | None = None, bold: bool = True) -> Text:
    """Colour each character along a multi-stop gradient."""
    stops = stops or GRADIENT
    rgb = [_hex_to_rgb(s) for s in stops]
    out = Text()
    n = max(len(text) - 1, 1)
    segments = len(rgb) - 1
    for i, ch in enumerate(text):
        pos = (i / n) * segments
        idx = min(int(pos), segments - 1)
        t = pos - idx
        r = _lerp(rgb[idx][0], rgb[idx + 1][0], t)
        g = _lerp(rgb[idx][1], rgb[idx + 1][1], t)
        b = _lerp(rgb[idx][2], rgb[idx + 1][2], t)
        out.append(ch, style=f"bold #{r:02x}{g:02x}{b:02x}" if bold else f"#{r:02x}{g:02x}{b:02x}")
    return out


def header(subtitle: str = "know what runs — before you download") -> Panel:
    """A compact branded header panel."""
    title = Text("✈ ", style=CYAN) + gradient_text("peyk")
    body = Group(Align.center(title), Align.center(Text(subtitle, style=MUTED)))
    return Panel(body, box=ROUNDED, border_style=BORDER, padding=(0, 2))


def dot(color: str) -> Text:
    return Text("●", style=color)


# Speed thresholds (tok/s) -> colour, à la a traffic light.
def speed_color(tps: float) -> str:
    if tps < 4:
        return RED
    if tps < 10:
        return YELLOW
    if tps < 30:
        return GREEN
    return "#22c55e"  # brisk


ACCEL_ICON = {"APPLE": "", "NVIDIA": "▲", "AMD": "△", "NONE": "▫"}


def accel_icon(accelerator: str) -> str:
    return ACCEL_ICON.get(accelerator, "▫")
