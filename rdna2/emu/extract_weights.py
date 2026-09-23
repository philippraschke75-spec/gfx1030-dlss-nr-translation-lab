"""Dump every WEIGHTS_HT record out of nvngx_dlssnr.dll into rdna2/build/weights/.

The container parser lives in weights_ht.py; this only does naming and file writing. Two names are
written per single-layer block so both existing conventions resolve: `blockN.bin` (what the encoder
and decoder chain runners read) and `blockN_layerM.bin` (what the C=512 and ViT runners read).

Output goes under rdna2/build/, which .gitignore excludes - these are vendor weight bytes and must
never enter the repository.

usage: py rdna2/emu/extract_weights.py <nvngx_dlssnr.dll>
"""
import sys, re, mmap, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import weights_ht

dll = Path(sys.argv[1])
out = Path(__file__).resolve().parent.parent / 'build' / 'weights'
out.mkdir(parents=True, exist_ok=True)

with open(dll, 'rb') as fh:
    m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    recs = weights_ht.records(m)

    layers = collections.defaultdict(list)
    for r in recs:
        g = re.match(r'block(\d+)\.layer(\d+)\.layer$', r['name'])
        if g:
            layers[int(g[1])].append(int(g[2]))

    written = total = 0
    for r in recs:
        g = re.match(r'block(\d+)\.layer(\d+)\.(\w+)$', r['name'])
        if not g:
            print('SKIP unparsed record name:', r['name'])
            continue
        blk, lay, kind = int(g[1]), int(g[2]), g[3]
        data = m[r['file_offset']:r['file_offset'] + r['bytes']]
        names = ['block%d_layer%d.bin' % (blk, lay)] if kind == 'layer' else \
                ['block%d_layer%d_%s.bin' % (blk, lay, kind)]
        if kind == 'layer' and len(layers[blk]) == 1:
            names.append('block%d.bin' % blk)       # single-layer blocks keep the short name
        for n in names:
            (out / n).write_bytes(data)
            written += 1
        total += r['bytes']

print('wrote %d files (%d records, %.1f MiB) to %s' % (written, len(recs), total / 2**20, out))

have = {int(x) for x in re.findall(r'block(\d+)_layer0\.bin', ' '.join(p.name for p in out.iterdir()))}
missing = [b for b in range(71) if b not in have]
print('blocks 0-70 with a layer0 file: %d, missing: %s' % (len(have), missing or 'none'))
