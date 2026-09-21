"""Package original research sources and audit evidence, excluding vendor bytes."""
from pathlib import Path
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import zipfile

root = Path(__file__).resolve().parent
out = root / 'handover'
repo = root / 'handover-upstream'
evidence = out / 'evidence'
evidence.mkdir(exist_ok=True)
names = [
    'alignment_test.cpp', 'fragment_test.cpp', 'fragment_wmma.h',
    'final_head_test.cpp', 'translate_final_head.py', 'run_fragment_test.ps1',
    'inspect_kernel.py', 'kernel_inventory.py', 'README.md', 'FINAL_HEAD_RESULTS.md',
    'alignment-run.log', 'fragment-run.log', 'final-head-abi.log',
    'final-head-abi-fixed.log', 'final-head-constant.log',
    'final-head-initialized.log', 'final-head-patterned.log',
    'final-head-multigroup.log', 'final-head-artifact-hashes.json',
]
dest = out / 'support' / 'rdna2'
dest.mkdir(parents=True, exist_ok=True)
for name in names:
    shutil.copy2(root / 'rdna2' / name, dest / name)
shutil.copy2(root / 'inspect_bundles.py', out / 'support' / 'inspect_bundles.py')
test = subprocess.run([sys.executable, '-m', 'unittest', 'discover',
                       '-s', 'tests/host', '-t', 'tests/host', '-v'],
                      cwd=repo, capture_output=True, text=True)
(evidence / 'host-tests.txt').write_text(test.stdout + test.stderr, encoding='utf-8')
if test.returncode:
    raise SystemExit('Host tests failed; do not issue passing report')
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
status = subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True)
manifest = json.loads((root / 'analysis/installer-bundles/manifest.json').read_text())
installer = Path(manifest['input'])
blob = installer.read_bytes()
base = 930304
assert blob[base:base+24] == b'__CLANG_OFFLOAD_BUNDLE__'
count = struct.unpack_from('<Q', blob, base+24)[0]
assert count == 9
assert hashlib.sha256(blob).hexdigest() == manifest['sha256']
inventory = json.loads((root / 'analysis/kernel-inventory.json').read_text())
report = {
    'audit_date': '2026-09-20', 'upstream_commit': head,
    'upstream_status_short': status, 'host_tests_exit_code': test.returncode,
    'gpu_tests_run_during_audit': False, 'game_run_during_audit': False,
    'installer_sha256': manifest['sha256'],
    'bundle_offset': base, 'u64_at_offset_24_entry_count': count,
    'targets': [x['target'] for x in manifest['entries'] if x['bundle_offset'] == base],
    'missing_public_inputs': {s: not (repo/s).exists() for s in [
        'src/bridge/bridge_gfx1030_fatbin.h', 'phase9_final_module/asm',
        'phase5_exact_fragment/gfx1100_code_object.o']},
    'static_function_count': len(inventory['kernels']),
    'functions_with_wmma_sites': sum(x['wmma_sites'] > 0 for x in inventory['kernels']),
    'wmma_sites': inventory['wmma_variants'],
    'sources_of_claims': {
        'upstream_physical_noncompletion': 'Pinned STATUS.md; not rerun here',
        'local_final_head_pass': 'Copied local source, artifact identities and retained logs',
        'external_gta_log': 'User-supplied RX 6600 report, not independently reproduced',
    },
}
(evidence/'inspection.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'package-files.json')
allowed = {'.md', '.py', '.ps1', '.cpp', '.h', '.json', '.log', '.txt'}
assert all(p.suffix in allowed for p in files)
records = [{
    'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size,
    'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
} for p in files]
(evidence/'package-files.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
archive = root / 'gfx1030-GTA-first-frame-handover.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.rglob('*')):
        if p.is_file():
            z.write(p, Path('gfx1030-handover') / p.relative_to(out))
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                  'files': len(records)+1, 'inspection': report}, indent=2))
