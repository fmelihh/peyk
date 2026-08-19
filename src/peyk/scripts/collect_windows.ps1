# peyk deep hardware probe (Windows).
# Emits a single JSON object on stdout via CIM/WMI. No admin needed for these
# classes; Win32_PhysicalMemory exposes real DIMM speed and count.
$ErrorActionPreference = 'SilentlyContinue'

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$mem = @(Get-CimInstance Win32_PhysicalMemory)

$memType = $null
switch (($mem | Select-Object -First 1).SMBIOSMemoryType) {
    26 { $memType = 'DDR4' }
    34 { $memType = 'DDR5' }
    24 { $memType = 'DDR3' }
}
$memSpeed = ($mem | Select-Object -First 1).Speed
$dimms = ($mem | Where-Object { $_.Capacity -gt 0 }).Count

$gpus = @()
foreach ($v in Get-CimInstance Win32_VideoController) {
    $vram = $null
    if ($v.AdapterRAM -gt 0) { $vram = [math]::Round($v.AdapterRAM / 1GB, 1) }
    $vendor = 'OTHER'
    if ($v.Name -match 'NVIDIA') { $vendor = 'NVIDIA' }
    elseif ($v.Name -match 'AMD|Radeon') { $vendor = 'AMD' }
    $gpus += [ordered]@{ vendor = $vendor; name = $v.Name; vram_gb = $vram }
}

# nvidia-smi gives accurate VRAM (AdapterRAM caps at 4 GB for larger cards).
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $nv = @()
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | ForEach-Object {
        $parts = $_ -split ','
        if ($parts.Count -ge 2) {
            $nv += [ordered]@{ vendor = 'NVIDIA'; name = $parts[0].Trim();
                               vram_gb = [math]::Round([double]($parts[1].Trim()) / 1024, 1) }
        }
    }
    if ($nv.Count -gt 0) { $gpus = $nv }
}

$out = [ordered]@{
    cpu    = [ordered]@{ model = $cpu.Name; cores_physical = $cpu.NumberOfCores;
                         cores_logical = $cpu.NumberOfLogicalProcessors }
    memory = [ordered]@{ type = $memType; speed_mtps = $memSpeed; dimms_populated = $dimms }
    gpus   = $gpus
}
$out | ConvertTo-Json -Depth 5 -Compress
