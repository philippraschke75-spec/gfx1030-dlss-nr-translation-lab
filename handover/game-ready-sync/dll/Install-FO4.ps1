# Installs our HIP NR backend into Fallout 4 on top of the already installed janblade OptiScaler-DLSSNR (winmm.dll proxy).
# Backs up every file it overwrites to E:\dlss 5-backups\fallout4-<timestamp>\ and writes Uninstall-FO4.ps1 there.
param(
  [string]$Game = "C:\Program Files (x86)\Steam\steamapps\common\Fallout 4",
  [string]$Build = "$PSScriptRoot\out",
  [string]$Runtime = "E:\dlss 5\handover\game-ready\package\nr_runtime.dll",
  [string]$DataDir = "E:\dlss-runtime\fo4_1280x720"
)
$ErrorActionPreference = "Stop"
if (Get-Process Fallout4 -ErrorAction SilentlyContinue) { throw "Fallout 4 is running - close it first." }
foreach ($p in "$Game\winmm.dll", "$Game\OptiScaler.ini", "$Build\nvngx.dll_dlssnr.dll", $Runtime, "$DataDir\manifest.txt") {
  if (-not (Test-Path $p)) { throw "missing: $p" } }

$bak = "E:\dlss 5-backups\fallout4-" + (Get-Date -Format "yyyyMMdd-HHmmss")
New-Item -ItemType Directory -Force $bak | Out-Null
$touched = "nvngx.dll_dlssnr.dll", "nr_runtime.dll", "nr_port.ini", "OptiScaler.ini"
$added = @()
foreach ($f in $touched) {
  if (Test-Path "$Game\$f") { Copy-Item "$Game\$f" "$bak\$f" } else { $added += $f } }
Set-Content "$bak\added.txt" ($added -join "`r`n") -Encoding ascii
Set-Content "$bak\game.txt" $Game -Encoding ascii

Copy-Item "$Build\nvngx.dll_dlssnr.dll" "$Game\nvngx.dll_dlssnr.dll" -Force
Copy-Item $Runtime "$Game\nr_runtime.dll" -Force
@"
; RDNA2 HIP build of nvngx.dll_dlssnr.dll (our port). Log: dlssnr_nr.log next to this file.
[nr]
; Model size must equal display size x WorkingScale (2560x1440 x 0.5 = 1280x720). Other sizes are refused (native frame stays).
data_dir=$DataDir
vit_every=1
log=dlssnr_nr.log
settle_ms=20
job_timeout_ms=3000
; apply=1: live frame + NR edit from an older frame (no lag). gain 0..1 scales the edit.
apply=1
gain=1.0
gate=1
gate_lo=0.03
gate_hi=0.15
"@ | Set-Content "$Game\nr_port.ini" -Encoding ascii

# NR only runs on OptiScaler's D3D11->D3D12 bridge: FSR 3.1 via dx11on12 instead of native-DX11 FSR 3.1.
$ini = Get-Content "$Game\OptiScaler.ini"
$ini = $ini -replace '^\s*Dx11Upscaler\s*=.*$', 'Dx11Upscaler = ffx_12'
Set-Content "$Game\OptiScaler.ini" $ini -Encoding utf8

@"
`$ErrorActionPreference = "Stop"
if (Get-Process Fallout4 -ErrorAction SilentlyContinue) { throw "Fallout 4 is running - close it first." }
`$bak = "$bak"; `$game = "$Game"
foreach (`$f in "nvngx.dll_dlssnr.dll", "nr_runtime.dll", "nr_port.ini", "OptiScaler.ini") { if (Test-Path "`$bak\`$f") { Copy-Item "`$bak\`$f" "`$game\`$f" -Force } }
foreach (`$f in (Get-Content "`$bak\added.txt")) { if (`$f -and (Test-Path "`$game\`$f")) { Remove-Item "`$game\`$f" -Force } }
Write-Host "restored from `$bak"
"@ | Set-Content "$bak\Uninstall-FO4.ps1" -Encoding utf8
Write-Host "installed. backup + uninstaller: $bak\Uninstall-FO4.ps1"
