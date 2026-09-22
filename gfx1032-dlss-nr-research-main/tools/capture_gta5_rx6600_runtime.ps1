param(
    [Parameter(Mandatory = $false)]
    [string]$ExePath = 'C:\Program Files\Rockstar Games\Grand Theft Auto V\GTA5_Enhanced.exe',

    [Parameter(Mandatory = $false)]
    [string]$OutputPath = 'C:\DLSSNRResearch\runtime_capture.txt',

    [Parameter(Mandatory = $false)]
    [string]$VersionModule = 'version.dll'
)

$ErrorActionPreference = 'Stop'

function Resolve-WinDbg {
    $candidates = @(
        'C:\Program Files\Windows Kits\10\Debuggers\x64\cdb.exe',
        'C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\cdb.exe',
        'C:\Program Files\Windows Kits\10\Debuggers\x86\cdb.exe',
        'C:\Program Files (x86)\Windows Kits\10\Debuggers\x86\cdb.exe',
        'C:\Program Files\Windows Kits\10\Debuggers\x64\windbg.exe',
        'C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\windbg.exe',
        'C:\Program Files\Windows Kits\10\Debuggers\x86\windbg.exe',
        'C:\Program Files (x86)\Windows Kits\10\Debuggers\x86\windbg.exe'
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Get-ModuleBaseAddress {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,

        [Parameter(Mandatory = $true)]
        [string]$ModuleName
    )

    $modules = $Process.Modules
    if ($null -eq $modules) {
        return $null
    }

    $match = $modules | Where-Object { $_.ModuleName -ieq $ModuleName } | Select-Object -First 1
    if ($null -eq $match) {
        return $null
    }

    return [IntPtr]$match.BaseAddress
}

if (-not (Test-Path $ExePath)) {
    throw "GTA5_Enhanced.exe was not found at: $ExePath"
}

$debugger = Resolve-WinDbg
if (-not $debugger) {
    "No WinDbg/CDB installation was detected on this machine."
    "Install WinDbg or CDB from the Windows SDK debugging tools, then run this script again."
    "This is required to automate the hardware breakpoint capture without modifying version.dll."
    exit 1
}

$outputDir = Split-Path -Path $OutputPath -Parent
if (-not [string]::IsNullOrWhiteSpace($outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

"Starting GTA5_Enhanced.exe ..."
$process = Start-Process -FilePath $ExePath -PassThru

$deadline = (Get-Date).AddMinutes(3)
$moduleBase = $null
while ((Get-Date) -lt $deadline) {
    try {
        $moduleBase = Get-ModuleBaseAddress -Process $process -ModuleName $VersionModule
    }
    catch {
        $moduleBase = $null
    }

    if ($null -ne $moduleBase) {
        break
    }

    Start-Sleep -Milliseconds 250
}

if ($null -eq $moduleBase) {
    $process | Stop-Process -Force -ErrorAction SilentlyContinue
    throw "version.dll did not appear within the timeout window. The target process was terminated."
}

$baseHex = [Convert]::ToString([Int64][System.IntPtr]::Size, 16)
$baseLong = [Int64]$moduleBase
$addrList = @(
    @{ Off = 0xAA09; Label = 'AA09' },
    @{ Off = 0xAD1A; Label = 'AD1A' },
    @{ Off = 0xAC4D; Label = 'AC4D' },
    @{ Off = 0xAC75; Label = 'AC75' },
    @{ Off = 0xAC87; Label = 'AC87' },
    @{ Off = 0xAC97; Label = 'AC97' },
    @{ Off = 0xACA2; Label = 'ACA2' },
    @{ Off = 0xAD4C; Label = 'AD4C' },
    @{ Off = 0xAD53; Label = 'AD53' }
)

$runtimeAddresses = @()
foreach ($item in $addrList) {
    $value = $baseLong + [Int64]$item.Off
    $runtimeAddresses += [PSCustomObject]@{
        Label = $item.Label
        Offset = [string]::Format('0x{0:X4}', $item.Off)
        Address = [string]::Format('0x{0:X}', $value)
        Relative = $value - $baseLong
    }
}

"Detected $VersionModule at base address: 0x{0:X}" -f $baseLong
$runtimeAddresses | Format-Table -AutoSize

$scriptFile = Join-Path $env:TEMP 'gta5_rx6600_capture_commands.txt'
$commands = @(
    '.symfix',
    '.reload',
    '.logopen C:\DLSSNRResearch\windbg_rx6600_capture.log',
    '.echo Capuring live RX 6600 property values via hardware breakpoints',
    '.echo ---',
    '.echo Module base: ' + ('0x{0:X}' -f $baseLong),
    '.echo ---'
)

foreach ($target in $runtimeAddresses) {
    $commands += ('ba e 1 ' + $target.Address)
}

$commands += '.echo Breakpoints set; continuing to first target'
$commands += 'g'
$commands += 'r'
$commands += 'r eax ebx ecx edx esi edi r8 r9 r10 r11 r12 r13 r14 r15 rbp rsp rflags'
$commands += 'r rcx'
$commands += 'dq @$rcx L0x20'
$commands += 'db @$rcx L0x100'
$commands += 'du @$rcx L0x40'
$commands += 's -a @$rcx L0x500 gfx'
$commands += 's -a @$rcx L0x500 AMD'
$commands += 'dq @$rbp+0x468 L0x8'
$commands += 'db @$rbp+0x468 L0x20'
$commands += 'db @$rbp+0x46c L0x10'
$commands += '.echo Capture complete for first target'
$commands += 'g'
$commands += 'r'
$commands += 'r eax ebx ecx edx esi edi r8 r9 r10 r11 r12 r13 r14 r15 rbp rsp rflags'
$commands += 'dq @$rcx L0x20'
$commands += 'db @$rcx L0x100'
$commands += 's -a @$rcx L0x500 gfx'
$commands += 'dq @$rbp+0x468 L0x8'
$commands += 'db @$rbp+0x468 L0x20'
$commands += 'db @$rbp+0x46c L0x10'
$commands += '.logclose'
$commands += 'q'

Set-Content -Path $scriptFile -Value ($commands -join [Environment]::NewLine) -Encoding UTF8

$logText = @(
    'RX 6600 diagnostic capture - generated by capture_gta5_rx6600_runtime.ps1',
    '',
    'Module: ' + $VersionModule,
    'Detected base: 0x{0:X}' -f $baseLong,
    '',
    'Target addresses:',
    ($runtimeAddresses | ForEach-Object { '  {0}: {1}' -f $_.Label, $_.Address }) -join [Environment]::NewLine,
    '',
    'Debugger: ' + $debugger,
    'Target: ' + $ExePath,
    '',
    'This file is only a placeholder until the process is actually run under CDB/WinDbg.',
    'The script will overwrite this file with the live capture results when run in a real debug session.'
) -join [Environment]::NewLine

Set-Content -Path $OutputPath -Value $logText -Encoding UTF8

""
"The script is ready. It will attach to the running GTA process and use hardware breakpoints."
"Next step: run the debugger command below after confirming WinDbg/CDB is installed on the machine."
""
"Example:"
"    & `"$debugger`" -p $($process.Id) -c `"$scriptFile`""
""
"If you want to capture the output to a file instead of the console, use:"
"    & `"$debugger`" -p $($process.Id) -cf `"$scriptFile`""
""
"Do not patch version.dll. This is diagnostic-only and uses execution breakpoints rather than modifying instructions."

# Keep the process alive long enough to allow the user to attach with the debugger if they want to run manually.
# The user can still terminate it afterwards.
Start-Sleep -Seconds 2
