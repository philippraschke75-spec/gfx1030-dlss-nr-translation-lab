param(
    [switch]$CheckOnly,
    [switch]$ViaVortex,
    [string]$VortexPath = 'C:\Program Files\Vortex\Vortex.exe'
)
$ErrorActionPreference = 'Stop'
$captureManifestPath = Join-Path $PSScriptRoot 'build\capture-bootstrap\launch-manifest.json'
$captureManifest = Get-Content -LiteralPath $captureManifestPath -Raw | ConvertFrom-Json
foreach ($captureFile in $captureManifest.files) {
    $captureHash = (Get-FileHash -LiteralPath $captureFile.path -Algorithm SHA256).Hash
    if ($captureHash -ne $captureFile.sha256) { throw "Capture file changed: $($captureFile.path)" }
}
if (Get-Process Cyberpunk2077 -ErrorAction SilentlyContinue) { throw 'Close Cyberpunk before starting a capture session.' }
if ($ViaVortex) {
    if (-not (Test-Path -LiteralPath $VortexPath -PathType Leaf)) { throw "Vortex executable not found: $VortexPath" }
    if (Get-Process Vortex -ErrorAction SilentlyContinue) {
        throw 'Exit Vortex completely, including its system-tray icon, then run this launcher again. An already-running Vortex cannot inherit capture settings.'
    }
}
Write-Output 'Capture files verified. Research launches will be blocked; this is not a rendering test.'
if ($CheckOnly) { return }
$captureOutputRoot = Join-Path $PSScriptRoot 'build\game-captures'
New-Item -ItemType Directory -Path $captureOutputRoot -Force | Out-Null
$captureOutput = Join-Path $captureOutputRoot ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N') + '.json')
$captureOldMode = $env:DLSSNR_RESEARCH_CAPTURE
$captureOldFile = $env:DLSSNR_CAPTURE_FILE
try {
    $env:DLSSNR_RESEARCH_CAPTURE = '1'
    $env:DLSSNR_CAPTURE_FILE = $captureOutput
    if ($ViaVortex) {
        # A fresh Vortex process inherits these variables and passes them to
        # directly launched children. No persistent user/system env changes.
        $captureProcess = Start-Process -FilePath $VortexPath -WorkingDirectory (Split-Path -Parent $VortexPath) -PassThru
        $captureMethod = 'vortex'
    } else {
        $captureProcess = Start-Process -FilePath (Join-Path $captureManifest.game_directory 'Cyberpunk2077.exe') -WorkingDirectory $captureManifest.game_directory -PassThru
        $captureMethod = 'direct'
    }
    [pscustomobject]@{pid=$captureProcess.Id;method=$captureMethod;capture=$captureOutput;started=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $captureOutputRoot 'last-session.json')
    Write-Output "Capture session started. Expected output: $captureOutput"
    if ($ViaVortex) {
        Write-Output 'Now use the Play button for Cyberpunk in the Vortex window that just opened.'
        Write-Output 'No in-game capture button is required. Exit Vortex after the test to end capture mode.'
        Write-Output 'If Vortex redirects the launch through an already-running Steam/REDlauncher process, capture settings may not reach the game; check bootstrap.log.'
    }
    Write-Output 'The file appears only if a recognized SWIN launch is reached. Close the game after the diagnostic attempt.'
} finally {
    $env:DLSSNR_RESEARCH_CAPTURE = $captureOldMode
    $env:DLSSNR_CAPTURE_FILE = $captureOldFile
}
