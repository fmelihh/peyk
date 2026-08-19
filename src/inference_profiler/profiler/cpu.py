"""CPU feature-flag detection across platforms."""

from __future__ import annotations

import platform
import subprocess
from typing import List

INTERESTING = ("avx512", "avx2", "avx", "neon", "fma", "f16c")


def _read_proc_cpuinfo() -> str:
    try:
        with open("/proc/cpuinfo", "r") as fh:
            return fh.read().lower()
    except OSError:
        return ""


def _sysctl(key: str) -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=3
        )
        return out.stdout.lower()
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_flags() -> List[str]:
    system = platform.system()
    blob = ""
    if system == "Linux":
        blob = _read_proc_cpuinfo()
    elif system == "Darwin":
        # Apple Silicon reports ARM NEON implicitly; Intel Macs expose features.
        blob = _sysctl("machdep.cpu.features") + _sysctl("machdep.cpu.leaf7_features")
        if platform.machine() == "arm64":
            blob += " neon fp16"
    # Windows / unknown: return whatever the arch implies.
    elif platform.machine().lower() in ("arm64", "aarch64"):
        blob = "neon"

    found = [flag for flag in INTERESTING if flag in blob]
    return found
