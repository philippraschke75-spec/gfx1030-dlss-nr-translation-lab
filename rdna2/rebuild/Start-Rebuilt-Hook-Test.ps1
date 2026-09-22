param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
foreach ($name in @('Cyberpunk2077','REDEngineErrorReporter','Vortex')) {
    if (Get-Process $name -ErrorAction SilentlyContinue) { throw "Close $name fully before starting the rebuilt hook test." }
}
$out = Join-Path $PSScriptRoot '..\build\rebuild'
$manifest = Get-Content -LiteralPath (Join-Path $out 'installed.json') -Raw | ConvertFrom-Json
foreach ($file in $manifest.files) {
    if ((Get-FileHash -LiteralPath $file.path).Hash -ne $file.sha256) { throw "Changed test file: $($file.path)" }
}
if ($CheckOnly) { Write-Output 'Rebuilt hook test preflight passed'; return }
$names = @('DLSSNR_RESEARCH_CAPTURE','DLSSNR_CAPTURE_FILE','DLSSNR_REBUILD_PROBE','DLSSNR_REBUILD_LOG','DLSSNR_REBUILD_FRAME')
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
try {
    $env:DLSSNR_RESEARCH_CAPTURE = $null
    $env:DLSSNR_CAPTURE_FILE = $null
    $env:DLSSNR_REBUILD_PROBE = '1'
    $env:DLSSNR_REBUILD_LOG = Join-Path $out ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N') + '.log')
    $env:DLSSNR_REBUILD_FRAME = $env:DLSSNR_REBUILD_LOG + '.ppm'
    $process = Start-Process 'C:\Program Files\Vortex\Vortex.exe' -WorkingDirectory 'C:\Program Files\Vortex' -PassThru
    [pscustomobject]@{vortex_pid=$process.Id;log=$env:DLSSNR_REBUILD_LOG;started=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content (Join-Path $out 'last-session.json')
    Write-Output 'Press Play in Vortex. This tests DXGI observation only; neural rendering is unavailable.'
    Write-Output 'Once in-game, press F8 once to request a frame copy. Unknown/conflicting queues block the copy and are logged.'
    Write-Output "Log: $env:DLSSNR_REBUILD_LOG"
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$saved[$name],'Process') }
}
