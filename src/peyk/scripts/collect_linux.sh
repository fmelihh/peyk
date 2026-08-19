#!/usr/bin/env bash
# peyk deep hardware probe (Linux).
# Emits a single JSON object on stdout. Best-effort: every field falls back to
# null when the underlying tool is missing or lacks privileges.
#
# RAM type/speed/DIMM count come from `dmidecode`, which needs root. Run via
# `sudo` (peyk does this with --sudo) to unlock accurate memory bandwidth.
set -u

json_escape() { sed 's/\\/\\\\/g; s/"/\\"/g' <<<"${1:-}"; }

# --- CPU ---
cpu_model="$(awk -F: '/model name/{gsub(/^ +/,"",$2); print $2; exit}' /proc/cpuinfo 2>/dev/null)"
[ -z "$cpu_model" ] && cpu_model="$(lscpu 2>/dev/null | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')"
cores_phys="$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); c=$2} /^Socket\(s\)/{gsub(/ /,"",$2); s=$2} END{if(c&&s) print c*s}')"
[ -z "${cores_phys:-}" ] && cores_phys="$(nproc 2>/dev/null)"
cores_log="$(nproc --all 2>/dev/null)"
[ -z "$cores_log" ] && cores_log="$(getconf _NPROCESSORS_ONLN 2>/dev/null)"

# --- Memory (dmidecode; needs root) ---
mem_type=""; mem_speed=""; dimms=""
if command -v dmidecode >/dev/null 2>&1; then
  dmi="$(dmidecode -t memory 2>/dev/null)"
  if [ -n "$dmi" ]; then
    mem_type="$(awk -F: '/^\tType:/ && $2 !~ /Unknown|Other|None/ {gsub(/ /,"",$2); print $2; exit}' <<<"$dmi")"
    mem_speed="$(awk -F: '/Configured Memory Speed:|Memory Speed:|^\tSpeed:/ {v=$2; gsub(/[^0-9]/,"",v); if(v+0>0){print v; exit}}' <<<"$dmi")"
    dimms="$(awk -F: '/^\tSize:/ {if ($2 ~ /[0-9]+ *[GM]B/) c++} END{print c+0}' <<<"$dmi")"
  fi
fi

# --- GPUs ---
gpus="[]"
if command -v nvidia-smi >/dev/null 2>&1; then
  entries=""
  while IFS=, read -r name mem; do
    name="$(sed 's/^ *//; s/ *$//' <<<"$name")"; mem="$(tr -dc '0-9' <<<"$mem")"
    [ -z "$mem" ] && continue
    vram="$(awk "BEGIN{printf \"%.1f\", $mem/1024}")"
    entries="${entries}{\"vendor\":\"NVIDIA\",\"name\":\"$(json_escape "$name")\",\"vram_gb\":$vram},"
  done < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null)
  gpus="[${entries%,}]"
elif command -v rocm-smi >/dev/null 2>&1; then
  vram_bytes="$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2{print $2}' | tr -dc '0-9')"
  if [ -n "$vram_bytes" ]; then
    vram="$(awk "BEGIN{printf \"%.1f\", $vram_bytes/1073741824}")"
    gpus="[{\"vendor\":\"AMD\",\"name\":\"AMD GPU\",\"vram_gb\":$vram}]"
  fi
fi

# --- assemble JSON ---
q() { [ -n "${1:-}" ] && printf '"%s"' "$(json_escape "$1")" || printf 'null'; }
n() { [ -n "${1:-}" ] && printf '%s' "$1" || printf 'null'; }

cat <<EOF
{
  "cpu": {"model": $(q "$cpu_model"), "cores_physical": $(n "$cores_phys"), "cores_logical": $(n "$cores_log")},
  "memory": {"type": $(q "$mem_type"), "speed_mtps": $(n "$mem_speed"), "dimms_populated": $(n "$dimms")},
  "gpus": $gpus
}
EOF
