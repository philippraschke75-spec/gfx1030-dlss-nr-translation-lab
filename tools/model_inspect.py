"""Inspect the network inside an nvngx_dlssnr.dll, and diff two of them (read-only, offline).

    py -I tools/model_inspect.py <nvngx_dlssnr.dll> [--records] [--json out.json]
    py -I tools/model_inspect.py <new.dll> --diff <old.dll>
    py -I tools/model_inspect.py --selftest [<310.8 dll>]

Prints the DLL's sha256 and PE version, every WEIGHTS_HT record (name, bytes, container element count, the
layer family and widths inferred from the size, and the matrix/table split), the block structure and the
totals. With --diff it lists same / resized / added / removed records, compares the architectures and
classifies the change:
    A  same structure (block and layer layout, layer families), only widths changed  -> 1-2 days
    B  same building blocks, rearranged (other block counts, levels or order)        -> days to a week
    C  a record or record kind no known layer family explains (new building blocks)   -> 2-3 weeks

Family inference reproduces the per-layer payload layouts documented in dlssnr-pytorch/dlssnr/extract.py
(fused Swin C=32..256, split Swin 512, ViT 1024/4096). Matrices are e4m3 (1 byte per parameter), the
tables (residual scales, position bias, gains) f16/f32. Nothing is written except the optional --json index.
"""
import argparse
import hashlib
import json
import mmap
import re
import struct
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'rdna2' / 'emu'))
import weights_ht  # noqa: E402  (container parser; asserts the record framing)

NAME = re.compile(r'block(\d+)\.layer(\d+)\.(\w+)$')
WIDTHS = range(32, 2049, 32)

# ---- per-family payload sizes: (matrix parameters, table bytes) ------------------------------------------


def swin(c, c_in, c_out):
    heads, hidden = max(1, c // 32), 4 * c
    up, down = c_out < c_in, c_out > c_in
    mat = hidden * c + (c * hidden if heads == 1 else c * (hidden // heads) + c * c) + 3 * c * c + c * c
    mat += c_in * c_out if (up or down) else 0
    tab = (2 * c + 32) + heads * 8192 + max(16, 4 * heads)
    tab += (2 * c if c == 32 else 2 * c - 32) if up else 0
    tab += (2 * c + (16 if c == 32 else 0)) if down else 2 * c + 16
    return mat, tab


BODY32 = (3 * 32 * 32 + 32 * 32, 8192 + 16 + 80)
PRE = (2 * 128 * 32 + BODY32[0], 16 + 1024 + 80 + BODY32[1])
POST = (2 * 128 * 32 + BODY32[0], 16 + 3 * 64 + BODY32[1] + 1024)


def split(kind, s, v=0):
    heads = s // 32
    return {'ffwd': (2 * s * s, 0), 'proj': (s * s, 2 * s),
            'qkv_attn': (3 * s * s, heads * 8192 + 4 * heads),
            'final_head': (v * s, 16), 'dec_upsample': (s * v, 2 * s)}[kind]


def vit(kind, v, h=0):
    return {'expand': (h * v, 16), 'contract': (v * h, 2 * v), 'qkv': (3 * v * v, v // 8),
            'projection': (v * v, 2 * v)}[kind]


SHAPES = {  # e4m3 matrices [out, in] and f16 tables; h = C/32 heads, H = 4C
    'pre_block': 'mlp_expand[128,32] mlp_contract[32,128] input_lift f16[32,16] qkv[96,32] out_proj[32,32] bias f16[4096]',
    'post_block': 'mlp_expand[128,32] mlp_contract[32,128] qkv[96,32] out_proj[32,32] out_project f16[16,32] gains f16[32]x3',
    'swin': 'mlp_expand[4C,C] mlp_mid[C,4C/h] (h>1) mlp_contract[C,C] or [C,4C] (h=1) qkv[3C,C] out_proj[C,C] bias f16[h,4096]',
    'swin_down': 'as swin + resample[2C,C]',
    'swin_up': 'as swin + resample[C,2C] + upsample_gain f16[C]',
    'split_ffwd': 'grouped: 8 x (A[S/8,S], Q0[S/2,S/8], Q2[S/8,S/2]) packed as 2 x [S,S]',
    'split_proj': 'weight[S,S] skip_weight f16[S]',
    'split_qkv_attn': 'qkv[3S,S] bias f16[S/32,4096] head_scale f32[S/32]',
    'split_final_head': 'weight[V,S]',
    'dec_upsample': 'weight[S,V] skip_weight f16[S]',
    'vit_expand': 'weight[H,V]', 'vit_contract': 'weight[V,H] skip_weight f16[V]',
    'vit_qkv': 'q,k,v[V,V] temperature f32[V/32]', 'vit_projection': 'weight[V,V] skip_weight f16[V]',
    'vit_attention': 'scale f16[1]', 'blend_scale': 'f16[1]',
}


def _solve(size, fn):
    """Widths w (multiples of 32) with sum(fn(w)) == size."""
    return [w for w in WIDTHS if sum(fn(w)) == size]


# ---- reading a DLL ----------------------------------------------------------------------------------------


def pe_version(m):
    """'a.b.c.d' from VS_FIXEDFILEINFO, plus the ProductName / FileDescription strings if present."""
    out = {}
    for path, off, size in weights_ht.resource(m):
        if not path.startswith('id16/'):
            continue
        blob = bytes(m[off:off + size])
        i = blob.find(struct.pack('<I', 0xFEEF04BD))
        if i >= 0:
            ms, ls = struct.unpack_from('<II', blob, i + 8)
            out['file_version'] = '%d.%d.%d.%d' % (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)
        for key in ('ProductName', 'FileDescription', 'ProductVersion'):
            k = blob.find(key.encode('utf-16le') + b'\0\0')
            if k >= 0:
                start = k + 2 * len(key) + 2
                start += (-start) % 4
                end = blob.find(b'\0\0', start)
                while end >= 0 and (end - start) % 2:
                    end = blob.find(b'\0\0', end + 1)
                out[key] = blob[start:end].decode('utf-16le', 'replace')
    return out


def read_dll(path, res='WEIGHTS_HT'):
    """{'sha256', 'size', 'version', 'records': [{name, bytes, count, trailer, sha}]} for one DLL."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            recs = []
            for r in weights_ht.records(m, res):
                o, d = r['file_offset'], r['bytes']
                type_id, layout_id, ndim = struct.unpack_from('<IIQ', m, o + d)
                recs.append(dict(name=r['name'], bytes=d, count=r['trailer_count'],
                                 container=dict(type=type_id, layout=layout_id, ndim=ndim),
                                 sha=hashlib.sha256(m[o:o + d]).hexdigest()[:16]))
            version = pe_version(m)
        finally:
            m.close()
    return dict(path=str(path), sha256=h.hexdigest(), size=Path(path).stat().st_size, version=version, records=recs)


# ---- inference ------------------------------------------------------------------------------------------


def analyse(records):
    """Annotate every record with family / widths / params and return (records, blocks, arch)."""
    recs = [dict(r) for r in records]
    layers = OrderedDict()
    for r in recs:
        g = NAME.match(r['name'])
        r['block'], r['layer'], r['tail'] = (int(g[1]), int(g[2]), g[3]) if g else (None, None, r['name'])
        if g and g[3] == 'layer':
            layers.setdefault(r['block'], []).append(r['layer'])
    blocks = OrderedDict(sorted((b, sorted(ls)) for b, ls in layers.items()))
    by = {(r['block'], r['layer']): r for r in recs if r['tail'] == 'layer'}
    known = {}

    def put(r, family, mat_tab, **widths):
        r.update(family=family, widths=widths, matrix_params=mat_tab[0], table_bytes=mat_tab[1],
                 params=mat_tab[0] + mat_tab[1] // 2)

    def put_scalar(r, family):
        r.update(family=family, widths={}, matrix_params=0, table_bytes=r['bytes'], params=r['bytes'] // 2)

    # pass 1: everything whose family follows from the block's layer count and the size alone
    for b, ls in blocks.items():
        n = len(ls)
        for layer in ls:
            r = by[b, layer]
            size = r['bytes']
            if n == 1:
                if size == sum(PRE):
                    put(r, 'pre_block', PRE, C=32)
                elif size == sum(POST):
                    put(r, 'post_block', POST, C=32)
                else:
                    for kind, fn in (('swin', lambda w: swin(w, w, w)), ('swin_down', lambda w: swin(w, w, 2 * w)),
                                     ('swin_up', lambda w: swin(w, 2 * w, w))):
                        ws = _solve(size, fn)
                        if len(ws) == 1:
                            put(r, kind, fn(ws[0]), C=ws[0], heads=max(1, ws[0] // 32))
                            break
            elif n in (4, 5) and by[b, 0]['bytes'] % 2 == 0 and _solve(by[b, 0]['bytes'], lambda w: split('ffwd', w)):
                kinds = ['ffwd', 'proj', 'qkv_attn', 'proj', 'final_head']
                if layer < 4:
                    ws = _solve(size, lambda w: split(kinds[layer], w))
                    if len(ws) == 1:
                        put(r, 'split_' + kinds[layer], split(kinds[layer], ws[0]), S=ws[0])
                        known.setdefault('S', ws[0])
            elif n == 5:
                kinds = ['expand', 'contract', 'qkv', 'attention', 'projection']
                if layer == 3 and size == 2:
                    put_scalar(r, 'vit_attention')
                elif layer in (2, 4):
                    ws = _solve(size, lambda w: vit(kinds[layer], w))
                    if len(ws) == 1:
                        put(r, 'vit_' + kinds[layer], vit(kinds[layer], ws[0]), V=ws[0])
                        known.setdefault('V', ws[0])
    # pass 2: records that need a width established elsewhere (S for the final head, V for hidden and the decoder)
    S, V = known.get('S'), known.get('V')
    for (b, layer), r in by.items():
        if 'family' in r:
            continue
        size, n = r['bytes'], len(blocks[b])
        if S and V and size == sum(split('final_head', S, V)):
            put(r, 'split_final_head', split('final_head', S, V), S=S, V=V)
        elif S and V and n == 1 and size == sum(split('dec_upsample', S, V)):
            put(r, 'dec_upsample', split('dec_upsample', S, V), S=S, V=V)
        elif V and n == 5 and layer in (0, 1):
            kind = ('expand', 'contract')[layer]
            hs = [h for h in range(V, 8 * V + 1, 32) if sum(vit(kind, V, h)) == size]
            if len(hs) == 1:
                put(r, 'vit_' + kind, vit(kind, V, hs[0]), V=V, H=hs[0])
    for r in recs:
        if 'family' not in r:
            if r['tail'] == 'blend_scale' and r['bytes'] == 2:
                put_scalar(r, 'blend_scale')
            else:
                r.update(family='UNKNOWN', widths={}, matrix_params=0, table_bytes=r['bytes'], params=r['bytes'])
    return recs, blocks, architecture(recs, blocks)


def architecture(recs, blocks):
    """Encoder levels (width x blocks), split stage, ViT, decoder levels, and anything unexplained."""
    fam = {(r['block'], r['layer']): r for r in recs if r['tail'] == 'layer'}
    seq = []          # (stage, width) per block, in order
    for b, ls in blocks.items():
        f = fam[b, 0]['family']
        if f.startswith('swin') or f in ('pre_block', 'post_block'):
            seq.append(('swin', fam[b, 0]['widths'].get('C')))
        elif f.startswith('split_'):
            seq.append(('split', fam[b, 0]['widths'].get('S')))
        elif f.startswith('vit_'):
            seq.append(('vit', fam[b, 2]['widths'].get('V') if (b, 2) in fam else None))
        elif f == 'dec_upsample':
            seq.append(('dec_upsample', fam[b, 0]['widths'].get('S')))
        else:
            seq.append(('UNKNOWN', None))
    runs = []
    for stage, w in seq:
        if runs and runs[-1][0] == stage and runs[-1][1] == w:
            runs[-1][2] += 1
        else:
            runs.append([stage, w, 1])
    hidden = sorted({r['widths']['H'] for r in recs if 'H' in r['widths']})
    return dict(stages=[tuple(x) for x in runs], vit_hidden=hidden,
                unknown=[r['name'] for r in recs if r['family'] == 'UNKNOWN'],
                families=dict(Counter(r['family'] for r in recs)))


def fmt_stages(arch):
    return ' -> '.join(f"{s}{'' if w is None else ' ' + str(w)}x{n}" for s, w, n in arch['stages'])


# ---- report ----------------------------------------------------------------------------------------------


def report(info, show_records=True):
    recs, blocks, arch = analyse(info['records'])
    print(f"file      {info.get('path', '-')}")
    if 'sha256' in info:
        print(f"sha256    {info['sha256']}  ({info['size']:,} bytes)")
        print(f"version   {info['version'] or '(no version resource)'}")
    if show_records:
        print(f"\n{'record':<30}{'bytes':>11}{'count':>10}  {'family':<18}{'widths':<22}{'matrix':>11}{'tables':>8}")
        for r in recs:
            w = ' '.join(f'{k}={v}' for k, v in r['widths'].items())
            print(f"{r['name']:<30}{r['bytes']:>11,}{r.get('count', 0):>10,}  {r['family']:<18}{w:<22}"
                  f"{r['matrix_params']:>11,}{r['table_bytes']:>8,}")
        print("  dtype: matrices e4m3 [rows, cols] (1 B/param); tables f16 (residual/skip scales, gains, 64x64 "
              "position bias per head) and f32 (head scales). Container element count = bytes/2 (nominal f16).")
        print("  shapes per family:")
        for f in sorted({r['family'] for r in recs}):
            if f in SHAPES:
                print(f"    {f:<18}{SHAPES[f]}")
    print(f"\nblocks    {len(blocks)} (block {min(blocks)}..{max(blocks)}), layers per block: "
          + ', '.join(f'{n}-layer x{c}' for n, c in sorted(Counter(len(v) for v in blocks.values()).items())))
    kinds = Counter(r['tail'] for r in recs)
    print(f"kinds     " + ', '.join(f'{k} x{v}' for k, v in sorted(kinds.items())))
    print(f"structure {fmt_stages(arch)}   (ViT hidden {arch['vit_hidden']})")
    print(f"families  " + ', '.join(f'{k} x{v}' for k, v in sorted(arch['families'].items())))
    mat = sum(r['matrix_params'] for r in recs)
    par = sum(r['params'] for r in recs)
    print(f"totals    {len(recs)} records, {sum(r['bytes'] for r in recs):,} bytes, "
          f"{par / 1e6:.2f}M parameters ({mat / 1e6:.2f}M e4m3 matrix weights + {(par - mat) / 1e6:.2f}M table values)")
    if arch['unknown']:
        print(f"UNKNOWN   {len(arch['unknown'])} records match no known layer family: {arch['unknown'][:8]}")
    return recs, blocks, arch


def classify(old, new):
    """('A'|'B'|'C'|'same', reason) from two analyse() results."""
    (orecs, oblocks, oarch), (nrecs, nblocks, narch) = old, new
    if narch['unknown'] or set(r['tail'] for r in nrecs) - set(r['tail'] for r in orecs) \
            or set(narch['families']) - set(oarch['families']) - {'UNKNOWN'}:
        return 'C', 'new building blocks: records or record kinds no known layer family explains'
    if [r['name'] for r in orecs] == [r['name'] for r in nrecs] and \
            [r['family'] for r in orecs] == [r['family'] for r in nrecs]:
        if [r['bytes'] for r in orecs] == [r['bytes'] for r in nrecs]:
            if [r['sha'] for r in orecs] == [r['sha'] for r in nrecs]:
                return 'same', 'identical weights'
            return 'A', 'same structure and widths, retrained weights only (drop-in)'
        return 'A', 'same structure, other widths (narrower/wider layers)'
    return 'B', 'known building blocks, rearranged (block counts, levels or order differ)'


def diff(old_info, new_info):
    print('=== OLD ===')
    old = report(old_info, show_records=False)
    print('\n=== NEW ===')
    new = report(new_info, show_records=False)
    o = {r['name']: r for r in old[0]}
    n = {r['name']: r for r in new[0]}
    shape = lambda r: f"{r['family']} {' '.join(f'{k}={v}' for k, v in r['widths'].items())} ({r['bytes']:,} B)"
    same = [k for k in o if k in n and o[k]['bytes'] == n[k]['bytes']]
    resized = [k for k in o if k in n and o[k]['bytes'] != n[k]['bytes']]
    removed = [k for k in o if k not in n]
    added = [k for k in n if k not in o]
    print(f"\n=== DIFF ===\nsame size {len(same)} (identical bytes {sum(o[k].get('sha') == n[k].get('sha') for k in same)}), "
          f"resized {len(resized)}, removed {len(removed)}, added {len(added)}")
    order = lambda k: tuple(int(x) for x in re.findall(r'\d+', k)) or (1 << 30,)
    resized, removed, added = (sorted(x, key=order) for x in (resized, removed, added))
    for k in resized:
        print(f"  resized  {k:<28} {shape(o[k])}  ->  {shape(n[k])}")
    for k in removed:
        print(f"  removed  {k:<28} {shape(o[k])}")
    for k in added:
        print(f"  added    {k:<28} {shape(n[k])}")
    print(f"\nstructure old: {fmt_stages(old[2])}   hidden {old[2]['vit_hidden']}")
    print(f"structure new: {fmt_stages(new[2])}   hidden {new[2]['vit_hidden']}")
    for a, b in zip(old[2]['stages'], new[2]['stages']):
        if a != b:
            print(f"  changed  {a[0]} {a[1]}x{a[2]} -> {b[0]} {b[1]}x{b[2]}")
    po, pn = sum(r['params'] for r in old[0]), sum(r['params'] for r in new[0])
    print(f"parameters {po / 1e6:.2f}M -> {pn / 1e6:.2f}M ({100 * (pn - po) / po:+.1f} %)")
    verdict, why = classify(old, new)
    print(f"\nSCENARIO {verdict}: {why}")
    return verdict


# ---- synthetic models (self-test) -------------------------------------------------------------------------


def synthetic(levels=((32, 4), (64, 4), (128, 6), (256, 8)), s=512, nsplit=8, v=1024, h=4096, nvit=8):
    """The record list (name, bytes) a model of this shape would carry; the default is 310.8's shape."""
    recs, b = [], 0

    def add(layer, size):
        recs.append(dict(name=f'block{b}.layer{layer}.layer', bytes=size, count=size // 2, sha=f'{b}.{layer}.{size}'))
    add(0, sum(PRE)); b += 1
    ends = []
    for c, count in levels:
        for i in range(count):
            add(0, sum(swin(c, c, 2 * c if i == count - 1 else c))); b += 1
        ends.append(b - 1)
    for i in range(nsplit):
        for layer, k in enumerate(('ffwd', 'proj', 'qkv_attn', 'proj')):
            add(layer, sum(split(k, s)))
        if i == nsplit - 1:
            add(4, sum(split('final_head', s, v)))
        b += 1
    for i in range(nvit):
        for layer, size in enumerate((sum(vit('expand', v, h)), sum(vit('contract', v, h)), sum(vit('qkv', v)), 2,
                                      sum(vit('projection', v)))):
            add(layer, size)
        b += 1
    add(0, sum(split('dec_upsample', s, v))); b += 1
    for i in range(nsplit):
        for layer, k in enumerate(('ffwd', 'proj', 'qkv_attn', 'proj')):
            add(layer, sum(split(k, s)))
        b += 1
    for c, count in reversed(levels):
        for i in range(count):
            add(0, sum(swin(c, 2 * c if i == 0 else c, c))); b += 1
    add(0, sum(POST))
    recs.append(dict(name=f'block{b}.layer0.blend_scale', bytes=2, count=1, sha='blend'))
    return sorted(recs, key=lambda r: r['name'])


def selftest(dll=None):
    base = synthetic()
    if dll:
        real = read_dll(dll)
        got = sorted((r['name'], r['bytes']) for r in real['records'])
        assert got == sorted((r['name'], r['bytes']) for r in base), 'synthetic 310.8 does not match the DLL'
        print('synthetic 310.8 record list == real DLL record list (%d records)' % len(got))
        base = real['records']
    _, _, arch = analyse(base)
    assert not arch['unknown'], arch['unknown']
    assert fmt_stages(arch) == ('swin 32x5 -> swin 64x4 -> swin 128x6 -> swin 256x8 -> split 512x8 -> vit 1024x8 -> '
                                'dec_upsample 512x1 -> split 512x8 -> swin 256x8 -> swin 128x6 -> swin 64x4 -> swin 32x5'), \
        fmt_stages(arch)
    narrow = synthetic(levels=((32, 4), (64, 4), (96, 6), (192, 8)), s=384, v=768, h=3072)
    rearranged = synthetic(levels=((32, 4), (64, 4), (128, 4), (256, 6)), nsplit=6, nvit=6)
    novel = synthetic()
    novel[10] = dict(novel[10], bytes=novel[10]['bytes'] + 4096)
    novel.append(dict(name='block3.layer0.lora_delta', bytes=4096, count=2048, sha='x'))
    for name, model, want in (('narrow', narrow, 'A'), ('rearranged', rearranged, 'B'), ('novel', novel, 'C')):
        verdict = classify(analyse(base), analyse(model))[0]
        print(f'{name:<11} -> scenario {verdict}')
        assert verdict == want, (name, verdict)
    print('\n--- example diff, 310.8 vs synthetic narrow model ---')
    diff(dict(path='310.8', records=base), dict(path='synthetic narrow', records=narrow))
    print('\nselftest ok')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dll', nargs='?')
    ap.add_argument('--diff', metavar='OLD_DLL', help='compare against this (older) DLL')
    ap.add_argument('--records', action='store_true', help='also list every record (default without --diff)')
    ap.add_argument('--json', help='write the analysed record index (no weights) to this file')
    ap.add_argument('--res', default='WEIGHTS_HT', help='weights resource of DLL (WEIGHTS_CC_SMALL_FP8 = 310.9 small model)')
    ap.add_argument('--diff-res', default='WEIGHTS_HT', help='weights resource of the --diff DLL')
    ap.add_argument('--selftest', action='store_true', help='check inference/classification on synthetic models')
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.dll)
    if not a.dll:
        ap.error('a DLL path is required')
    new = read_dll(a.dll, a.res)
    if a.diff:
        verdict = diff(read_dll(a.diff, a.diff_res), new)
        result = dict(new=new, scenario=verdict)
    else:
        recs, blocks, arch = report(new, show_records=True)
        result = dict(new=dict(new, records=recs), arch=arch)
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1, default=str))


if __name__ == '__main__':
    main()
