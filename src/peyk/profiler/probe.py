"""Deep hardware probe: shell out to platform scripts for richer info.

The baseline `detect()` uses portable Python libraries (psutil). This module
goes further by running bundled shell / PowerShell scripts that call native
tools (dmidecode, lscpu, system_profiler, CIM) to recover details those
libraries don't expose — most importantly RAM type/speed/DIMM count, which lets
peyk compute a *measured* memory bandwidth instead of a table estimate.

Everything is best-effort: a missing tool, denied sudo, or malformed output
leaves the baseline profile untouched.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from importlib import resources

from ..models import Accelerator, HardwareProfile
from .bandwidth import estimate_bandwidth

_SCRIPTS = {
    "Linux": "collect_linux.sh",
    "Darwin": "collect_macos.sh",
    "Windows": "collect_windows.ps1",
}

# One 64-bit channel at N MT/s moves N * 8 bytes/s. GB/s = dimms * MT/s * 8 / 1000.
_BYTES_PER_CHANNEL = 8


def _script_command(script_path: str, allow_sudo: bool) -> list[str]:
    system = platform.system()
    if system == "Windows":
        return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path]
    prefix: list[str] = []
    # sudo only helps on Linux (dmidecode); skip if already root.
    if allow_sudo and system == "Linux" and os.geteuid() != 0:
        prefix = ["sudo", "bash"]
        return prefix + [script_path]
    return ["bash", script_path]


def run_probe(allow_sudo: bool = False, timeout: float = 20.0) -> dict | None:
    """Run the platform probe script and return parsed JSON, or None."""
    script = _SCRIPTS.get(platform.system())
    if not script:
        return None
    try:
        with resources.as_file(
            resources.files("peyk").joinpath("scripts").joinpath(script)
        ) as path:
            # stdout is captured for JSON; stderr/stdin inherit the tty so an
            # interactive `sudo` password prompt still works.
            proc = subprocess.run(
                _script_command(str(path), allow_sudo),
                stdout=subprocess.PIPE,
                timeout=timeout,
                text=True,
            )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except (ValueError, TypeError):
        return None


def bandwidth_from_dimms(dimms: int | None, speed_mtps: int | None) -> float | None:
    if not dimms or not speed_mtps:
        return None
    return round(dimms * speed_mtps * _BYTES_PER_CHANNEL / 1000, 1)


def enrich_profile(profile: HardwareProfile, data: dict | None) -> HardwareProfile:
    """Fold deep-probe data into a baseline profile. Pure and side-effect free."""
    if not data:
        return profile

    cpu = data.get("cpu") or {}
    if cpu.get("model"):
        profile.cpu_model = str(cpu["model"]).strip()
        # The chip name is the accelerator name on Apple Silicon; refresh it so
        # the bandwidth lookup can distinguish e.g. M3 Max from a base M3.
        if profile.accelerator == Accelerator.APPLE:
            profile.accelerator_name = profile.cpu_model

    mem = data.get("memory") or {}
    if mem.get("type"):
        profile.ram_type = str(mem["type"])
    if mem.get("speed_mtps"):
        profile.ram_speed_mtps = int(mem["speed_mtps"])
    if mem.get("dimms_populated"):
        profile.ram_channels = int(mem["dimms_populated"])

    # Prefer a measured bandwidth for CPU-only hosts; otherwise re-estimate with
    # whatever better accelerator name we just learned.
    measured = None
    if profile.accelerator == Accelerator.NONE:
        measured = bandwidth_from_dimms(profile.ram_channels, profile.ram_speed_mtps)
    if measured:
        profile.mem_bandwidth_gbs = measured
        profile.mem_bandwidth_source = "measured"
    else:
        profile.mem_bandwidth_gbs = estimate_bandwidth(profile)
        profile.mem_bandwidth_source = "estimated"

    return profile
