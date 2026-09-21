"""Stage an instrumented bridge copy; never edit upstream or install a DLL."""
import hashlib
from pathlib import Path
import shutil
import argparse
import json
import subprocess
import re

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = '3dd21a0ea290761180453a03d80d570035f4219f40b1b278bd15811ff405ea97'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-only', action='store_true', help='no substitution; permanently block every launch')
    parser.add_argument('--output', type=Path, default=ROOT / 'rdna2/build/launch-capture-bridge')
    parser.add_argument('--backend', type=Path, default=Path('C:/Program Files/AMD/ROCm/6.4/bin/amdhip64_6.dll'))
    parser.add_argument('--build', action='store_true')
    options = parser.parse_args()
    src = ROOT / 'research/src/bridge'
    raw = (src / 'amdhip64_7.cpp').read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('bridge source changed; review insertion points before staging')
    out = options.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file(): shutil.copy2(p, out / p.name)
    shutil.copy2(ROOT / 'rdna2/launch_capture.h', out / 'launch_capture.h')
    text = raw.decode()
    # Record entry before backend resolution/call, so a crash inside an API
    # is distinguishable from a call that was never reached.
    anchor = '        using Fn = hipError_t (*) params;'
    if text.count(anchor) != 1: raise ValueError('forwarding macro changed')
    text = text.replace(anchor, '        log_line("ENTER %s", #name);                                          \\\n' + anchor)
    def insert(anchor, addition):
        nonlocal text
        if text.count(anchor) != 1: raise ValueError('ambiguous insertion point')
        text = text.replace(anchor, addition + anchor)
    insert('#include "registry_ident.h"', '#include "launch_capture.h"\n')
    extra = ('BRIDGE_FWD_ERR(hipEventCreateWithFlags, (hipEvent_t* event, unsigned flags), (event, flags))\n'
             'BRIDGE_FWD_ERR(hipEventQuery, (hipEvent_t event), (event))\n'
             'BRIDGE_FWD_ERR(hipStreamCreateWithFlags, (hipStream_t* stream, unsigned flags), (stream, flags))\n'
             'BRIDGE_FWD_ERR(hipStreamSynchronize, (hipStream_t stream), (stream))\n\n')
    insert('BRIDGE_FWD_ERR(__hipPopCallConfiguration,', extra)
    insert('    log_line("__hipRegisterFunction module=',
           '    launch_capture::registration(modules, hostFunction, deviceName);\n')
    insert('    log_line("__hipUnregisterFatBinary handle=',
           '    launch_capture::unregister_module(modules);\n')
    insert('    using Fn = hipError_t (*)(const void*, hipDim3, hipDim3, void**, size_t,',
           '    const unsigned capture_grid[] = {numBlocks.x,numBlocks.y,numBlocks.z};\n'
           '    const unsigned capture_block[] = {dimBlocks.x,dimBlocks.y,dimBlocks.z};\n'
           '    if (launch_capture::before_launch(function_address, capture_grid, capture_block,\n'
           '                                      args, sharedMemBytes, stream))\n'
           '        return blocked_launch("CAPTURE_ONLY", function_address,\n'
           '                              numBlocks.x,numBlocks.y,numBlocks.z);\n')
    if options.capture_only:
        text = text.replace('#include "registry_ident.h"', '// Capture-only: no registry substitution dependency.')
        start = text.index('    if (data != nullptr) {')
        end = text.index('    void** handle = fn(fwd);', start)
        text = text[:start] + '    // Original payload forwarded unchanged; never execute its kernels.\n' + text[end:]
        insert('    using Fn = hipError_t (*)(const void*, hipDim3, hipDim3, void**, size_t,',
               '    return blocked_launch("CAPTURE_ONLY_ALWAYS_BLOCKED", function_address,\n'
               '                          numBlocks.x,numBlocks.y,numBlocks.z);\n')
    backend = options.backend.resolve()
    if not backend.is_file(): raise ValueError('backend DLL missing')
    (out / 'bridge_config.h').write_text('#pragma once\n#define DLSSNR_BACKEND_PATH L' + json.dumps(str(backend)) + '\n')
    exports = (src / 'exports.def').read_text()
    for name in ('hipEventCreateWithFlags', 'hipEventQuery', 'hipStreamCreateWithFlags', 'hipStreamSynchronize'):
        exports += '    ' + name + '\n'
    (out / 'exports.def').write_text(exports)
    (out / 'amdhip64_7.cpp').write_text(text)
    if options.build:
        if not options.capture_only:
            raise ValueError('automated build is limited to the independent capture-only variant')
        subprocess.run(['C:/Program Files/AMD/ROCm/6.4/bin/hipcc.exe', '--offload-host-only', '-std=c++17', '-shared',
                        str(out / 'amdhip64_7.cpp'), '-o', str(out / 'amdhip64_7.dll')], check=True)
        readobj = 'C:/Program Files/AMD/ROCm/6.4/bin/llvm-readobj.exe'
        def exported(path):
            output = subprocess.check_output([readobj, '--coff-exports', str(path)], text=True)
            return set(re.findall(r'^\s+Name: (\S+)', output, re.M))
        expected = {line.strip() for line in exports.splitlines()
                    if line.strip() and not line.lstrip().startswith(';') and line.strip() != 'EXPORTS'}
        actual, backend_exports = exported(out/'amdhip64_7.dll'), exported(backend)
        if actual != expected or not expected <= backend_exports:
            raise ValueError('export mismatch: DLL=%s backend_missing=%s' %
                             (sorted(actual ^ expected), sorted(expected-backend_exports)))
        print('Verified all %d bridge exports and backend exports' % len(expected))
    (out / 'capture-build.json').write_text(json.dumps(dict(capture_only=options.capture_only,
        backend=str(backend), source_sha256=EXPECTED, payload_substitution=not options.capture_only,
        dll_sha256=hashlib.sha256((out/'amdhip64_7.dll').read_bytes()).hexdigest() if options.build else None), indent=2))
    print(out)
    print('Header dependency available:', (out / 'bridge_gfx1030_fatbin.h').exists())


if __name__ == '__main__': main()
