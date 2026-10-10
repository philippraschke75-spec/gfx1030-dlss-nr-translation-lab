"""Run every emulator bit-exactness check (CPU only) and write OUTDIR/evidence.json.
usage: run_all.py BUILD OUTDIR"""
import sys, json, subprocess, struct
from pathlib import Path
D = Path(__file__).resolve().parent
build, outdir = sys.argv[1], Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
CASES = Path('E:/dlss 5/astra-work/bounded-tests/cases')
C32 = '_Z10k_swin_varILi32ELb0EEv9VarParams'
jobs = []
for c in sorted(CASES.glob('c32_f*')):
    if (c / 'kernarg.bin').exists():
        lds = ['65536', 'hw'] if '_f8_' in c.name else [None]
        jobs += [(['case', build, C32, str(c)] + ([l] if l else []), '%s lds=%s' % (c.name, l)) for l in lds]
# real captured frame: all 8 c32 dispatches (incl. flag 8) and every other translated dispatch
man = [l.split('|') for l in Path('E:/dlss-vit/g_A/manifest.txt').read_text().splitlines() if l]
ka = Path('E:/dlss-vit/g_A/kernargs.bin').read_bytes()
for sym, n in [(C32, i) for i in range(8)] + [('_Z12k_attention212AttnParams1d', 0), ('_Z12k_attention212AttnParams1d', 7),
               ('_Z14k_dec_upsample11DecUpParams', 0), ('_Z12k_final_head10HeadParams', 0), ('_Z8k_import12ImportParams', 0),
               ('_Z16k_conv_res_views12ConvPlParams', 0), ('_Z8k_export12ExportParams', 0), ('_Z8k_repack12RepackParams', 0),
               ('_Z8k_repack12RepackParams', 1)]:
    row = [m for m in man if m[1] == sym][n]
    tag = 'frame %s #%d' % (sym, n)
    if sym == C32:
        tag += ' flags=%d' % struct.unpack_from('<I', ka, int(row[2]) + 0x28)[0]
    jobs.append((['frame', build, sym, str(n)], tag))
res = {}
for args, tag in jobs:
    print('RUN', tag, flush=True)
    env = dict(__import__('os').environ)
    if args[0] == 'frame' and 'flags=8' in tag:   # flag 8 overruns its declared LDS on the real frame too
        env['EMU_LDS'] = 'hw'; tag += ' lds=hw'
    r = subprocess.run([sys.executable, '-B', str(D / 'emu_check.py')] + args, capture_output=True, text=True, timeout=3600, env=env)
    last = (r.stdout.strip().splitlines() or [''])[-1]
    ok = r.returncode == 0 and last.startswith('PASS')
    res[tag] = {'pass': ok, 'result': json.loads(last[5:]) if ok else (r.stdout[-1500:] + r.stderr[-1500:])}
    print('  ', 'PASS' if ok else 'FAIL', flush=True)
    (outdir / 'evidence.json').write_text(json.dumps(res, indent=1))
print('passed %d/%d' % (sum(v['pass'] for v in res.values()), len(res)))
