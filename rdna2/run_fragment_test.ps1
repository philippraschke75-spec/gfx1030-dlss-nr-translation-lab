param([ValidateSet('fragment', 'alignment', 'final_head', 'kernels', 'conv_res', 'lowering_probe')][string]$Test = 'fragment',
      [ValidateSet('abi','constant','patterned','multigroup','basis')][string]$Mode = 'abi',
      [switch]$BuildOnly)
$ErrorActionPreference = 'Stop'
$HipRoot = 'C:\Program Files\AMD\ROCm\6.4'
$Build = Join-Path $PSScriptRoot 'build'
New-Item -ItemType Directory -Force -Path $Build | Out-Null
$VsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$VsLocation = & $VsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsLocation) { throw 'MSVC build tools unavailable' }
Import-Module (Join-Path $VsLocation 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -InstallPath $VsLocation -SkipAutomaticLocation -Arch amd64 -HostArch amd64 -DevCmdArguments '-no_logo'
$env:HIP_PATH = $HipRoot
$env:HIP_PLATFORM = 'amd'
$env:Path = "$HipRoot\bin;$env:Path"
$Source = Join-Path $PSScriptRoot "${Test}_test.cpp"
$Exe = Join-Path $Build "${Test}_test.exe"
& "$HipRoot\bin\clang++.exe" -x hip --offload-arch=gfx1030 "--hip-path=$HipRoot" -O3 -std=c++17 $Source -o $Exe
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed; no GPU test run' }
Get-FileHash -LiteralPath $Exe -Algorithm SHA256 | Format-List
if ($BuildOnly) { return }
if ($Test -in @('kernels','conv_res','lowering_probe')) { throw 'Use -BuildOnly, then run the executable with an explicit module and fixture.' }
if ($Test -eq 'final_head') {
    $ModuleName = if ($Mode -eq 'abi') { 'abi_probe.co' } else { 'final_head.co' }
    $ModulePath = Join-Path $Build "final-head\$ModuleName"
    Get-FileHash -LiteralPath $ModulePath -Algorithm SHA256 | Format-List
    & $Exe $ModulePath $Mode
} else {
    & $Exe
}
if ($LASTEXITCODE -ne 0) { throw "GPU test failed with exit $LASTEXITCODE; no retry" }
