"""Isolated test sandbox.

make:  python sandbox.py make <sandbox_dir>
        Copies the committed sources (rdna2/*.py|*.cpp, rdna2/emu/*.py) and only the local inputs the
        tools need into <sandbox_dir>, mirroring the repo layout the scripts expect. Nothing is written
        back to the repository.
run:   python sandbox.py run <sandbox_dir> <timeout_s> -- <command...>
        Runs a command with cwd inside the sandbox, a hard wall-clock limit and a kill of the whole
        process tree on expiry. Used for every GPU dispatch.

Limits (stated so nobody over-trusts this): the sandbox isolates files and process lifetime. It does
not isolate the GPU. A hung kernel can still make the Windows driver reset the device, and this tool
never changes watchdog/TDR settings.
"""
import os, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
INPUTS = [                                           # local, non-published inputs the tools read
    'analysis/gfx1100-disassembly.txt',
    'analysis/gfx1100-notes.txt',
    'analysis/installer-bundles/000e3200-3-hipv4-amdgcn-amd-amdhsa--gfx1100.elf',
]
NEEDED_PY = ['p14d_kd.py', 'p14d_dec.py']            # parsers imported from the upstream checkout


def make(dest):
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / 'rdna2' / 'emu').mkdir(parents=True)
    for p in (REPO / 'rdna2').iterdir():
        if p.suffix in ('.py', '.cpp', '.h'):
            shutil.copy2(p, dest / 'rdna2' / p.name)
    for p in (REPO / 'rdna2' / 'emu').iterdir():
        if p.suffix == '.py':
            shutil.copy2(p, dest / 'rdna2' / 'emu' / p.name)
    for rel in INPUTS:
        src = REPO / rel
        if not src.exists():
            print('MISSING local input:', rel)
            continue
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / rel)
    # parsers: prefer the pinned upstream checkout, fall back to the research/ copy
    for base in ('handover-upstream', 'research'):
        d = REPO / base / 'src' / 'emulator'
        if all((d / n).exists() for n in NEEDED_PY):
            (dest / 'research' / 'src' / 'emulator').mkdir(parents=True, exist_ok=True)
            for n in NEEDED_PY:
                shutil.copy2(d / n, dest / 'research' / 'src' / 'emulator' / n)
            break
    print('sandbox ready:', dest)


def run(dest, timeout, cmd):
    proc = subprocess.Popen(cmd, cwd=dest, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
    t0 = time.time()
    while proc.poll() is None:
        if time.time() - t0 > timeout:
            print('SANDBOX TIMEOUT after %.0f s: killing process tree' % timeout, flush=True)
            if os.name == 'nt':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], capture_output=True)
            else:
                proc.kill()
            return 124
        time.sleep(0.2)
    return proc.returncode


if __name__ == '__main__':
    if sys.argv[1] == 'make':
        make(sys.argv[2])
    elif sys.argv[1] == 'run':
        i = sys.argv.index('--')
        sys.exit(run(sys.argv[2], float(sys.argv[3]), sys.argv[i + 1:]))
