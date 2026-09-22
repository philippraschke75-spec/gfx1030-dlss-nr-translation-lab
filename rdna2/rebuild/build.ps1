$ErrorActionPreference = 'Stop'
$compiler = 'C:\Program Files\AMD\ROCm\6.4\bin\clang.exe'
$out = Join-Path $PSScriptRoot '..\build\rebuild'
New-Item -ItemType Directory -Path $out -Force | Out-Null
$objects = @()
foreach ($source in @('buffer.c','hook.c','trampoline.c','hde/hde64.c')) {
    $object = Join-Path $out ((Split-Path $source -Leaf) + '.obj')
    & $compiler -c -O2 (Join-Path $PSScriptRoot "third_party/minhook/src/$source") -o $object
    if ($LASTEXITCODE) { throw "Compile failed: $source" }
    $objects += $object
}
& $compiler -std=c++17 -O2 -shared (Join-Path $PSScriptRoot 'graphics_hook.cpp') @objects -o (Join-Path $out 'DLSSNRGraphicsProbe.dll')
if ($LASTEXITCODE) { throw 'DLL build failed' }
& $compiler -std=c++17 -O2 (Join-Path $PSScriptRoot 'graphics_hook_test.cpp') -o (Join-Path $out 'graphics_hook_test.exe')
if ($LASTEXITCODE) { throw 'Test build failed' }
Write-Output "Built diagnostic hook layer in $out"
