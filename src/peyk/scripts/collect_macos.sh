#!/usr/bin/env bash
# peyk deep hardware probe (macOS).
# Emits a single JSON object on stdout. Uses sysctl + system_profiler; no root
# needed. The key win here is the exact Apple chip name (e.g. "Apple M3 Max"),
# which lets peyk look up the correct unified-memory bandwidth.
set -u

json_escape() { sed 's/\\/\\\\/g; s/"/\\"/g' <<<"${1:-}"; }

# Prefer the marketing chip name from system_profiler; fall back to sysctl.
chip="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip:/{print $2; exit}')"
[ -z "$chip" ] && chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"

cores_log="$(sysctl -n hw.logicalcpu 2>/dev/null)"
cores_phys="$(sysctl -n hw.physicalcpu 2>/dev/null)"

# Memory type/speed (Intel Macs report speed; Apple Silicon is unified/LPDDR).
mem_type="$(system_profiler SPMemoryDataType 2>/dev/null | awk -F': ' '/Type:/{print $2; exit}')"
mem_speed_raw="$(system_profiler SPMemoryDataType 2>/dev/null | awk -F': ' '/Speed:/{print $2; exit}')"
mem_speed="$(tr -dc '0-9' <<<"${mem_speed_raw:-}")"

q() { [ -n "${1:-}" ] && printf '"%s"' "$(json_escape "$1")" || printf 'null'; }
n() { [ -n "${1:-}" ] && printf '%s' "$1" || printf 'null'; }

cat <<EOF
{
  "cpu": {"model": $(q "$chip"), "cores_physical": $(n "$cores_phys"), "cores_logical": $(n "$cores_log")},
  "memory": {"type": $(q "$mem_type"), "speed_mtps": $(n "$mem_speed"), "dimms_populated": null},
  "gpus": []
}
EOF
