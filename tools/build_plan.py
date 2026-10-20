"""Build the DLSS-NR frame plan (nr_data/<W>x<H>/) from nvngx_dlssnr.dll alone.

    py -I tools/build_plan.py <nvngx_dlssnr.dll> <WxH>[,<WxH>...] <out nr_data dir> [--kernels DIR] [--no-flex]
    --res WEIGHTS_CC_SMALL_FP8 picks the 310.9 small model (default WEIGHTS_HT = the 310.8 model, also in 310.9)
    --hybrid F[@L]: pre/post (and U-net levels < L) at WxH, levels >= L at (W/F)x(H/F), k_resample in between (see plan())
    py -I tools/build_plan.py --selftest [nvngx_dlssnr.dll]      (synthetic shrunk models, no GPU, no output)

Inputs: the WEIGHTS_HT records of the DLL (names + sizes; the bytes only for blend_scale and the weights.map hash),
the kernel dir (native .co modules + the translated .co files still in use) and rdna2/plan/nr_params.h (the kernarg
structs). Output per resolution: manifest.txt, kernargs.bin, overrides.txt, meta.txt/meta.json, weights.map,
arena.lzms (all zero: weights come from the user's DLL at load) and kernels/. Same consumer as before (nr_runtime).

Architecture: derived from the record sizes here, and cross-checked against dlssnr-pytorch-flex's Arch.from_weights
when its venv exists (--no-flex skips that). Anything the derivation or the kernels cannot cover stops with PlanError
naming the record or the missing kernel. See handover/own-plan.md.
"""
import math, mmap, os, re, struct, sys, json, shutil, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HEADER = REPO / 'rdna2' / 'plan' / 'nr_params.h'
FLEX = Path(r'E:/dlss 5/dlssnr-pytorch-flex'); FLEX_PY = FLEX.parent / 'dlssnr-pytorch' / '.venv' / 'Scripts' / 'python.exe'
DEFAULT_KERNELS = Path(r'E:/dlss-runtime/c32fix/nr_data/960x540/kernels')
BASE = 0x7500_0000_0000      # virtual arena base in kernargs.bin; nr_runtime rebases every qword in [BASE, BASE+arena)
ALIGN = 1 << 16              # buffer alignment in the arena
SLOT = 424                   # NR_KA_SLOT
RES = 'WEIGHTS_HT'           # weights resource; set by --res
MODES = [(0, 0), (-4, -4), (-4, 0), (0, -4)]   # (ox, oy) window origin by block index within a level


class PlanError(Exception):
    pass


# ------------------------------------------------------------------ nr_params.h
_TY = {'uint64_t': ('Q', 8), 'int32_t': ('i', 4), 'uint32_t': ('I', 4), 'float': ('f', 4), 'uint16_t': ('H', 2)}


def parse_header(path=HEADER):
    """{struct: (size, {field: (offset, fmt, count)})} from nr_params.h, checked against its static_asserts."""
    src = re.sub(r'//[^\n]*', '', Path(path).read_text())
    out = {}
    for name, body in re.findall(r'struct\s+(\w+)\s*\{(.*?)\};', src, re.S):
        off, fields = 0, {}
        for decl in body.split(';'):
            decl = decl.strip()
            if not decl: continue
            m = re.fullmatch(r'(?:const\s+)?([\w]+)\s*(\*?)\s*(.+)', decl, re.S)
            ty, ptr, names = m[1], m[2], m[3]
            for nm in names.split(','):
                nm = nm.strip(); star = ptr or nm.startswith('*'); nm = nm.lstrip('* ')
                cnt = 1; a = re.fullmatch(r'(\w+)\[(\d+)\]', nm)
                if a: nm, cnt = a[1], int(a[2])
                fmt, sz = ('Q', 8) if star else _TY[ty]
                off = (off + sz - 1) // sz * sz
                fields[nm] = (off, fmt, cnt, bool(star)); off += sz * cnt
        size = (off + 7) // 8 * 8 if any(f[1] == 'Q' for f in fields.values()) else off
        want = re.search(r'sizeof\(%s\)\s*==\s*(0x[0-9a-fA-F]+|\d+)' % name, src)
        if not want or int(want[1], 0) != size: raise PlanError('nr_params.h: %s parses to %d bytes, static_assert says %s' % (name, size, want and want[1]))
        out[name] = (size, fields)
    return out


# ------------------------------------------------------------------ weights
def dll_records(dll):
    sys.path.insert(0, str(REPO / 'rdna2' / 'emu')); import weights_ht
    f = open(dll, 'rb'); m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    recs = {r['name']: (r['file_offset'], r['bytes']) for r in weights_ht.records(m, RES)}
    return recs, m, f


def _cc2(n):   # n = C*C + 2*C -> C
    c = math.isqrt(n + 1) - 1
    return c if c * c + 2 * c == n else None


def derive(size):
    """Architecture from {record name: bytes}. Rules hold for 310.8; a model breaking one raises PlanError."""
    L = {}
    for nm in size:
        m = re.fullmatch(r'block(\d+)\.layer(\d+)\.(layer|blend_scale)', nm)
        if not m: raise PlanError('unknown record %r (new record family: see handover/new-model-day1.md step C)' % nm)
        if m[3] == 'layer': L.setdefault(int(m[1]), {})[int(m[2])] = size[nm]
    blocks = sorted(L); post = blocks[-1]
    if blocks != list(range(post + 1)): raise PlanError('block numbers not contiguous: missing %s' % sorted(set(range(post + 1)) - set(blocks)))
    if 'block%d.layer0.blend_scale' % post not in size: raise PlanError('no blend_scale on the last block %d' % post)
    single = lambda b: list(L[b]) == [0]
    vit = [b for b in blocks if 4 in L[b] and L[b].get(3) == 2]
    if not vit: raise PlanError('no ViT blocks (layer4 present, layer3 = 2 bytes)')
    mid = [b for b in blocks if 1 in L[b]]
    c1 = [b for b in mid if b < vit[0]]; decup = vit[-1] + 1; c2 = [b for b in mid if b > decup]
    if not (c1 and c2 and c1 == list(range(c1[0], vit[0])) and vit == list(range(vit[0], decup)) and single(decup)
            and c2 == list(range(decup + 1, c2[-1] + 1)) and 4 in L[c1[-1]]):
        raise PlanError('mid section is not <C_mid enc> <ViT> <dec_upsample> <C_mid dec> (blocks %s / %s / %s)' % (c1, vit, c2))
    for b in c1 + c2:
        if sorted(L[b]) != ([0, 1, 2, 3, 4] if b == c1[-1] else [0, 1, 2, 3]): raise PlanError('C_mid block %d has layers %s' % (b, sorted(L[b])))
    c_mid = _cc2(L[c1[0]][1]); c_vit = _cc2(L[vit[0]][4])
    if not c_mid or not c_vit: raise PlanError('C_mid/C_vit: layer1 / ViT layer4 are not C*C+2C bytes')
    hid = (L[vit[0]][0] - 16) // c_vit
    heads = (L[c1[0]][2] - 3 * c_mid * c_mid - 256 * c_mid) // 4
    chk = {(c1[0], 0): 2 * c_mid * c_mid, (c1[0], 2): 3 * c_mid * c_mid + 256 * c_mid + 4 * heads, (c1[-1], 4): c_mid * c_vit + 16,
           (vit[0], 0): hid * c_vit + 16, (vit[0], 1): hid * c_vit + 2 * c_vit, (vit[0], 2): 3 * c_vit * c_vit + 128, (decup, 0): c_mid * c_vit + 2 * c_mid}
    for (b, l), n in chk.items():
        if L[b][l] != n: raise PlanError('record block%d.layer%d.layer is %d bytes, the derived widths (C_mid %d, C_vit %d, hidden %d) give %d' % (b, l, L[b][l], c_mid, c_vit, hid, n))
    for b in vit + c1 + c2:   # same widths everywhere in a run
        ref = c1[0] if b in c1 + c2 else vit[0]
        for l in L[b]:
            if l < 4 and L[b][l] != L[ref][l]: raise PlanError('block%d.layer%d differs from block%d (mixed widths inside a run)' % (b, l, ref))
    rec = lambda b: L[b][0]

    def runs(bl, closing):   # encoder: an odd record closes a level; decoder: an odd record opens one
        st, cur = [], []
        for b in bl:
            if not single(b): raise PlanError('Swin block %d has %d layers' % (b, len(L[b])))
            if closing:
                cur.append(b)
                if len(cur) > 1 and rec(b) != rec(cur[0]): st.append(cur); cur = []
            else:
                if len(cur) > 1 and rec(b) != rec(cur[1]): st.append(cur); cur = []
                cur.append(b)
        if cur: st.append(cur)
        return st
    enc = runs(range(1, c1[0]), True); dec = runs(range(c2[-1] + 1, post), False)
    n = len(enc)
    if len(dec) != n: raise PlanError('encoder has %d levels, decoder %d' % (n, len(dec)))
    if [len(s) for s in dec] != [len(s) for s in enc][::-1]: raise PlanError('decoder level sizes %s do not mirror the encoder %s' % ([len(s) for s in dec], [len(s) for s in enc]))
    if c_mid % (1 << n): raise PlanError('C_mid %d not divisible by 2^%d' % (c_mid, n))
    return dict(n=n, c0=c_mid >> n, c_mid=c_mid, heads=heads, c_vit=c_vit, hid=hid, post=post, c1=c1, vit=vit, decup=decup, c2=c2,
                enc=[(c_mid >> (n - k), s) for k, s in enumerate(enc)], dec=[(c_mid >> (k + 1), s) for k, s in enumerate(dec)])


def flex_check(dll, A):
    """Cross-check against the MIT reference (dlssnr-pytorch-flex Arch.from_weights). Skipped if its venv is missing."""
    if not FLEX_PY.exists(): print('flex: venv not found, cross-check skipped'); return
    code = ('import json,sys,contextlib,io\nimport dlssnr.extract as ex\nex.RESOURCE_NAME = sys.argv[2]\nfrom dlssnr.extract import extract\nfrom dlssnr.plan import Arch\n'
            'with contextlib.redirect_stdout(io.StringIO()): w = extract(sys.argv[1], None)\n'
            'a = Arch.from_weights(w); print(json.dumps([a.levels, a.split_channels, a.split_blocks, a.vit_channels, a.vit_hidden, a.vit_blocks]))')
    r = subprocess.run([str(FLEX_PY), '-c', code, str(Path(dll).resolve()), RES], cwd=FLEX, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(FLEX), DLSSNR_TRITON='0'))
    if r.returncode: raise PlanError('dlssnr-pytorch-flex could not read the weights:\n' + r.stderr[-2000:])
    lv, sc, sb, vc, vh, vb = json.loads(r.stdout.strip().splitlines()[-1])
    mine = [[[c // 32 or 1, c, len(s)] for c, s in A['enc']], A['c_mid'], len(A['c1']), A['c_vit'], A['hid'], len(A['vit'])]
    if [lv, sc, sb, vc, vh, vb] != mine: raise PlanError('flex reference disagrees: flex %s, ours %s' % ([lv, sc, sb, vc, vh, vb], mine))
    print('flex: architecture agrees with dlssnr-pytorch-flex', mine)


# ------------------------------------------------------------------ kernels
def module_syms(path):
    try: return set(re.findall(rb'(k_[A-Za-z0-9_]+|_Z\w+)(?=\x00)', Path(path).read_bytes()))
    except OSError: return set()


def tsym(op, A, C=None):   # the translated (host ABI) symbol of an op
    return {'import': '_Z8k_import12ImportParams', 'pre': '_Z21k_pre_block_1h_32_fp89PreParams', 'post': '_Z22k_post_block_1h_32_fp810PostParams',
            'var': '_Z10k_swin_varILi%dELb0EEv9VarParams' % (C or 0), 'ffwd2': '_Z7k_ffwd211Ffwd2Params', 'conv2': '_Z11k_conv_res211Conv2Params',
            'attn': '_Z11k_qkv_attn210AttnParams', 'convpl': '_Z16k_conv_res_views12ConvPlParams', 'head': '_Z12k_final_head10HeadParams',
            'repack': '_Z8k_repack12RepackParams', 'expand': '_Z9k_expand212ExpandParams', 'contract': '_Z11k_contract212ConvParams1d',
            'qkv': '_Z6k_qkv29QkvParams', 'attn1d': '_Z12k_attention212AttnParams1d', 'decup': '_Z14k_dec_upsample11DecUpParams',
            'export': '_Z8k_export12ExportParams', 'mirror': 'k_mirror_pad', 'resample': 'k_resample'}[op]   # mirror, resample: native only


def native_candidates(op, A, C, grid):
    """Native kernels per op, in preference order: (module, symbol, grid, threads, why-not or None).
    New native kernels (e.g. the no-translated branch's import/export/repack/attention/head/conv_views) go here."""
    gx, gy, gz = grid
    M310 = (A['c_mid'], A['heads']) == (512, 16); V310 = (A['c_vit'], A['hid']) == (1024, 4096)
    if op == 'var':
        out = []
        if C == 32: out.append(('native_swin32.co', 'k_swin_c32', grid, 64, None))
        if C == 64: out.append(('native_swin64n.co', 'k_swin_c64', grid, 64, None))   # lane = token (swin_c64n.hip)
        if C >= 256: out.append(('native_swinvar.co', 'k_swin_c%ds' % C, (gx, gy, 2), 256, None))
        out.append(('native_swinvar.co', 'k_swin_c%d' % C, grid, 256, None if C % 32 == 0 else 'needs C % 32 == 0'))
        return out
    nat = {'mirror': ('native_mirror.co', 'k_mirror_pad', grid, 256, None),
           'resample': ('native_resample.co', 'k_resample', grid, 256, None),
           'pre': ('native_pre_native.co', 'k_pre_native', grid, 64, None if A['c0'] == 32 else 'hardwired C0 = 32'),
           'post': ('native_post_native.co', 'k_post_native', grid, 64, None if A['c0'] == 32 and A['post_ok'] else
                    'hardwired C0 = 32 and the 310.8 block%d record layout (k_post_native_prep reads it at load)' % A['post']),
           # native_io.co (no-translated). k_final_head_n / k_conv_res_views_n read their tile count T at kernarg 0x18 / 0x48,
           # which is the hidden block_count_x right after the struct, so their grid x must stay T (= the translated grid x).
           'import': ('native_io.co', 'k_import_n', grid, 256, None),
           'export': ('native_io.co', 'k_export_n', grid, 256, None),
           'repack': ('native_io.co', 'k_repack_n', grid, 256, None),
           'attn1d': ('native_io.co', 'k_attention_n', grid, 256, None if V310 else 'hardwired 1024'),
           'head': ('native_io.co', 'k_final_head_n', (gx, 16, gz), 256, None if M310 and V310 else 'hardwired 512 -> 1024'),
           'convpl': ('native_io.co', 'k_conv_res_views_n', (gx, 8, gz), 256, None if M310 else 'hardwired C_mid 512'),
           'conv2': ('native_c512.co', 'k_conv2_n', (gx, A['c_mid'] // 64, gz), 256, None if M310 else 'hardwired C_mid 512'),
           'ffwd2': ('native_c512.co', 'k_ffwd2_n', grid, 256, None if M310 else 'hardwired C_mid 512'),
           'attn': ('native_c512.co', 'k_qkv_attn_n', grid, 256, None if M310 else 'hardwired C_mid 512 / 16 heads'),
           # ViT GEMMs: 128 threads (vit_swz.hip, 8x4 register tile per thread)
           'expand': ('native_vit.co', 'k_vit_expand_n', (gx, A['hid'] // 64, gz), 128, None if V310 else 'hardwired 1024/4096'),
           'contract': ('native_vit.co', 'k_vit_contract_n', (gx, A['c_vit'] // 64, gz), 128, None if V310 else 'hardwired 1024/4096'),
           'qkv': ('native_vit.co', 'k_vit_qkv_n', (gx, 3 * A['c_vit'] // 64, gz), 128, None if V310 else 'hardwired 1024'),
           'decup': ('native_decup.co', 'k_dec_upsample_n', (gx, 4, gz), 256, None if M310 and V310 else 'hardwired 512 <- 1024')}
    return [nat[op]] if op in nat else []


def translated_ok(op, A, C):
    """The translated kernels are NVIDIA's 310.8 code: only valid at the 310.8 widths they were compiled for."""
    if op == 'var': return None
    if op in ('import', 'export'): return None
    if op in ('pre', 'post'): return None if A['c0'] == 32 else 'compiled for C0 = 32'
    if op in ('ffwd2', 'conv2', 'attn', 'convpl'): return None if (A['c_mid'], A['heads']) == (512, 16) else 'compiled for C_mid 512 / 16 heads'
    return None if (A['c_mid'], A['c_vit'], A['hid']) == (512, 1024, 4096) else 'compiled for 512 / 1024 / 4096'


def choose(op, A, C, grid, kdir, cache):
    """-> (manifest module, manifest sym, grid, threads, override or None). Natives go straight into the
    manifest; nr_runtime (own-plan or later) finds pre/post by meta pre_step/post_step, so no override lines are emitted."""
    syms = lambda f: cache.setdefault(f, module_syms(kdir / f))
    ts = tsym(op, A, C); tfile = ts + '.co'
    t_ok = (syms(tfile) and ts.encode() in syms(tfile)) and translated_ok(op, A, C) is None
    tried = []
    for mod, sym, g, thr, why in native_candidates(op, A, C, grid):
        if why: tried.append('native %s (%s)' % (sym, why)); continue
        if sym.encode() not in syms(mod): tried.append('native %s (not in %s)' % (sym, kdir / mod)); continue
        return 'kernels\\' + mod, sym, g, thr, None
    if t_ok: return 'kernels\\' + tfile, ts, grid, 256, None
    tried.append('translated %s (%s)' % (ts, translated_ok(op, A, C) or 'not in %s' % kdir))
    raise PlanError('no kernel for %s%s: %s' % (op, '' if C is None else ' C=%d' % C, '; '.join(tried)))


def geometry(W, H, levels):
    """Padded field and level extents as NVIDIA (OpenDLSS-NR Geometry::fromValid, = dlssnr-pytorch plan.padding/_walk):
    level[k] = alignUp(ceil(level[k-1] / 2), 4) from the field; the field is the valid size aligned to 2^reductions
    (floor 320), plus one alignment unit in x when both axes are 4-unit aligned. Returns (PH, PW, [(h, w) per level])."""
    up = lambda n, a: -(-n // a) * a

    def reductions(v):
        r, size = 0, v
        for level in range(levels):
            half = up(-(-size // 2), 4)
            if half < size: r += 1   # (an extra reduction when level 0 is not 8-aligned gave 1152 instead of 1088 at 1080p)
            size = half
        return r
    ah, aw = 1 << reductions(H), 1 << reductions(W)
    PH, PW = max(320, up(H, ah)), max(320, up(W, aw))
    if PW % (4 * aw) == 0 and PH % (4 * ah) == 0: PW += aw
    ext = [(PH, PW)]
    for _ in range(levels): ext.append((up(-(-ext[-1][0] // 2), 4), up(-(-ext[-1][1] // 2), 4)))
    return PH, PW, ext[1:]


# ------------------------------------------------------------------ plan
def plan(A, W, H, kdir, structs, hyb=None):
    """hyb = (F, L): hybrid resolution. pre/post and the U-net levels < L run on the W x H field, levels L.. (n = the C_mid
    bottleneck, ViT included) on the field of round(W/F) x round(H/F); k_resample (bilinear, planar e4m3) bridges the level-L
    input (pre half output / pool of level L-1) and the level-L decoder output. L = 0: the whole main net.
    Not the trained layout (NVIDIA runs every level at exactly half of the one above): a speed/look trade."""
    n = A['n']
    PH, PW, ext = geometry(W, H, n + 2)
    F_, L = hyb or (1, n + 2)
    WM, HM = round(W / F_), round(H / F_)
    mext = geometry(WM, HM, n + 2)[2]
    hw = lambda k: (mext if k >= L else ext)[k]                    # level k extent (n = C_mid stage, n + 1 = ViT)
    act = lambda C, h, w: C * -(-h // 4) * -(-w // 4) * 16 * 2      # 4x4-tile activation, 2 B/element (headroom)
    bufs, top = [], 0

    def buf(name, size):
        nonlocal top
        o = -(-top // ALIGN) * ALIGN; top = o + size; bufs.append((name, o, size)); return o
    src = buf('src RGBA16F', W * H * 8)
    rgb = buf('rgb RGB32F', PH * PW * 12)
    full = buf('pre out full', act(A['c0'], PH, PW))
    half = buf('pre out half', act(2 * A['c0'], *ext[0]))
    CL = A['c_mid'] >> (n - L) if hyb else 0                         # channels at the level-L input / decoder output
    if hyb: hin, hout = buf('hybrid in', act(CL, *mext[L])), buf('hybrid out', act(CL, *ext[L]))
    enc_b = [[buf('enc%d %s' % (k, s), act(C, *hw(k))) for s in ('ping', 'pong', 'pool')] for k, (C, _) in enumerate(A['enc'])]
    H5, W5 = hw(n); PH6, PW6 = (H5 + 1) // 2, (W5 + 1) // 2   # pooled extent convpl writes; the ViT extent rounds it up to 4
    H6, W6 = hw(n + 1)                                             # (the extra tokens are zero and take part in the attention)
    NM, NV = act(A['c_mid'], H5, W5), act(A['c_vit'], H5, W5)
    T5 = -(-H5 // 4) * -(-W5 // 4); T6 = -(-H6 // 4) * -(-W6 // 4); NTOK = -(-(H6 * W6) // 64) * 64
    m1 = [buf('mid enc w%d' % i, NM) for i in range(4)]
    vb = [buf('vit work%d' % i, NV) for i in range(6)]
    m2 = [buf('mid dec w%d' % i, NM) for i in range(4)]
    decin = buf('decoder input', NM)
    pooled = buf('pooled', T6 * A['c_mid'] * 16 * 2)
    headb = buf('head', T6 * A['c_vit'] * 16 * 2)
    vitres = buf('vit result', T6 * A['c_vit'] * 16 * 2)
    tok = [buf('vit tok%d' % i, NTOK * A['c_vit'] * 2) for i in range(2)]
    dec_b = [[buf('dec%d %s' % (k, s), act(C, *hw(n - 1 - k))) for s in ('ping', 'pong', 'pool')] for k, (C, _) in enumerate(A['dec'])]
    netout = buf('net result RGBA32F', PH * PW * 16)
    dst = buf('dst RGBA16F', W * H * 8)
    wofs = {nm: buf('w:' + nm, sz) for nm, sz in sorted(A['records'].items(), key=lambda t: [int(x) for x in re.findall(r'\d+', t[0])]) if not nm.endswith('blend_scale')}
    wl = lambda b, l: wofs['block%d.layer%d.layer' % (b, l)]

    steps = []   # (op, C, struct, {field: value}, grid)
    S = lambda op, st, f, grid, C=None: steps.append((op, C, st, f, tuple(grid) + (1,) * (3 - len(grid))))
    # native pre/post that export the marker k_<op>_native_fused_io do k_import / k_export themselves (two dispatches
    # less); older native and translated ones need the steps
    def native(op):
        try: mod, sym = choose(op, A, None, (1, 1, 1), kdir, {})[:2]
        except PlanError: return False
        return (sym + '_fused_io').encode() in module_syms(kdir / mod.split('\\')[-1])
    fuse = native('pre') and native('post')
    io = dict(src=src, src_h=H, src_w=W) if fuse else {}
    if not fuse:   # the fused native pre reads src mirrored itself (same reflection as k_mirror_pad)
        S('import', 'ImportParams', dict(src=src, src_pitch=W * 8, src_format=0, src_h=H, src_w=W, pad_h=PH, pad_w=PW, rgb=rgb, mode=0, scale=1.0), (-(-PW // 256), PH))
        S('mirror', 'MirrorParams', dict(rgb=rgb, h=H, w_=W, pad_h=PH, pad_w=PW), (-(-PW // 256), PH))
    S('pre', 'PreParams', dict(rgb=rgb, out_full=full, w=wl(0, 0), h=PH, w_=PW, in_scale=0.0625, tone=1.0, structure=1.0, out_half=half, a=1.0, b=1.0, **io), (PW // 8, PH // 8))   # NVIDIA's default conditioning (tone 1; auto mask: 1, skin 1, structure 1); runtime: nr_set_conditioning
    scratch = [None]
    rs = lambda a, b, sa, sb, f: S('resample', 'ResampleParams', dict(src=a, dst=b, sh=sa[0], sw=sa[1], dh=sb[0], dw=sb[1], planes=CL // 16, sx=f[0], sy=f[1]), (-(-sb[1] // 256), sb[0], CL // 16))

    def swin(C, k_hw, blocks, inp, pp, pool, dec, prev=None):
        h, w = k_hw; out = []
        start = 2 if dec and len(blocks) % 4 == 2 else 0   # a decoder level of 6 blocks starts its shift cycle at mode 2 (as dlssnr-pytorch-flex plan._rows)
        for i, b in enumerate(blocks):
            ox, oy = MODES[(start + i) % 4]; last = i == len(blocks) - 1
            flags = ((8 if i == 0 else 0) | (2 if last else 0)) if dec else ((1 if i == 0 else 0) | (4 if last else 0))
            f = dict(**{'in': inp if i == 0 else pp[(i + 1) % 2]}, out=pp[i % 2], w=wl(b, 0), H=h, W=w, X0=ox, Y0=oy, flags=flags)   # X0 = column origin (native c0 = X0 + 8 bx), matches the grid; the DLL capture had them swapped (window shift on the wrong axis, see handover/accuracy)
            if dec and i == 0: f['p30'] = prev
            else: f['p38'] = pool
            S('var', 'VarParams', f, ((w - ox + 7) // 8, (h - oy + 7) // 8), C)
            out.append(pp[i % 2])
        return out
    stage_in, skips = half, []
    for k, (C, blocks) in enumerate(A['enc']):
        if hyb and k == L: rs(stage_in, hin, ext[L], mext[L], (W / WM, H / HM)); stage_in = hin
        outs = swin(C, hw(k), blocks, stage_in, enc_b[k][:2], enc_b[k][2], False)
        skips.append(outs[-2] if len(outs) > 1 else stage_in)   # the decoder skip is the second-to-last block's output (host order)
        stage_in = enc_b[k][2]
    if any(len(b) < 2 for _, b in A['enc']): raise PlanError('an encoder level with 1 block has no defined skip buffer')

    def mid(blocks, work, first_in, enc):
        g4, g8 = (-(-T5 // 4), 4), (-(-T5 // 4), 8)
        for i, b in enumerate(blocks):
            first = enc and i == 0
            S('ffwd2', 'Ffwd2Params', dict(x=None if first else work[0], xp=first_in if first else None, out=work[1], w=wl(b, 0), H=H5, W=W5, T=T5), g8)
            S('conv2', 'Conv2Params', dict(x=work[1], res=None if first else work[0], resp=first_in if first else None, out=work[2], w=wl(b, 1), H=H5, W=W5, T=T5), g4)
            ox, oy = MODES[i % 4]
            S('attn', 'AttnParams', dict(x=work[2], out=work[3], w=wl(b, 2), H=H5, W=W5, ox=ox, oy=oy), ((W5 + 7 - ox) >> 3, (H5 + 7 - oy) >> 3, A['heads']))
            if enc and b == blocks[-1]:
                S('convpl', 'ConvPlParams', dict(a=work[3], res2=work[2], out=work[0], w=wl(b, 3), h=H5, w_=W5, pooled=pooled, ph=PH6, pw=PW6), (-(-NM // 16384), 1))
            else:
                S('conv2', 'Conv2Params', dict(x=work[3], res=work[2], out=work[0], out2=decin if (not enc and b == blocks[-1]) else None, w=wl(b, 3), H=H5, W=W5, T=T5), g4)
    if hyb and L == n: rs(stage_in, hin, ext[L], mext[L], (W / WM, H / HM)); stage_in = hin
    bn_lo = len(steps)
    mid(A['c1'], m1, stage_in, True)
    S('head', 'HeadParams', dict(pooled=pooled, out=headb, w=wl(A['c1'][-1], 4)), (T6, 1))
    vit_lo = len(steps)
    S('repack', 'RepackParams', dict(src=headb, dst=tok[0], h=H6, w_=W6, ntok=NTOK, to_tokens=1), (256, 1))
    vin, vout = tok
    B260, B268, B270, B278, B280, B288 = vb
    gy32, gy8 = (NTOK // 64, 32), (NTOK // 64, 8)
    for b in A['vit']:
        S('expand', 'ExpandParams', dict(x=vin, out=B260, w=wl(b, 0)), gy32)
        S('contract', 'ConvParams1d', dict(x=B260, skip=vin, out=B268, w=wl(b, 1), K=A['hid'], KN=A['hid'] * A['c_vit'], four=4), gy8)
        S('qkv', 'QkvParams', dict(x=B268, out=[B270, B278, B280], w=wl(b, 2)), gy32)
        S('attn1d', 'AttnParams1d', dict(q=B270, k=B278, v=B280, out=B288, ntok=NTOK, nreal=H6 * W6), gy32)
        S('contract', 'ConvParams1d', dict(x=B288, skip=B268, out=vout, w=wl(b, 4), K=A['c_vit'], KN=A['c_vit'] * A['c_vit'], four=4), gy8)
        vin, vout = vout, vin
    S('repack', 'RepackParams', dict(src=vin, dst=vitres, h=H6, w_=W6, ntok=NTOK, to_tokens=0), (256, 1))
    vit_hi = len(steps)
    S('decup', 'DecUpParams', dict(x=vitres, skip=m1[0], out=m2[0], w=wl(A['decup'], 0), H5=H5, W5=W5), (T5, 1))
    mid(A['c2'], m2, None, False)
    bn_hi = len(steps)
    stage_in = decin
    up = lambda: rs(stage_in, hout, mext[L], ext[L], (WM / W, HM / H)) or hout
    if hyb and L == n: stage_in = up()
    for k, (C, blocks) in enumerate(A['dec']):
        stage_in = swin(C, hw(n - 1 - k), blocks, skips[n - 1 - k], dec_b[k][:2], dec_b[k][2], True, prev=stage_in)[-1]
        if hyb and n - 1 - k == L: stage_in = up()
    blend = A['blend']
    S('post', 'PostParams', dict(x=stage_in, skip=full, out=netout, w=wl(A['post'], 0), h=PH, w_=PW, scale=0.03125, _r34=1, rgb=rgb, blend=blend, **(dict(dst=dst, dst_h=H, dst_w=W, src=src) if fuse else {})), (PW // 8, PH // 8))   # full padded frame: the margin feeds the next frame's history (cropping it changes history output)
    if not fuse: S('export', 'ExportParams', dict(res=netout, res_stride=PW, h=H, w_=W, dst_pitch=W * 8, dst=dst, mode=0, rgb=rgb, gain=1.0), (-(-W // 256), H))

    # kernels, then pack
    cache, lines, ovr, blob, mods = {}, [], {}, bytearray(), set()
    missing = []                 # every missing kernel at once, not just the first
    for op, C, st, f, grid in steps:
        try: choose(op, A, C, grid, kdir, cache)
        except PlanError as e:
            if str(e) not in missing: missing.append(str(e))
    if missing: raise PlanError('\n  '.join(missing))
    for op, C, st, f, grid in steps:
        if op == 'var' and not scratch[0] and translated_choice(op, A, C, grid, kdir, cache):
            scratch[0] = buf('swin scratch (translated k_swin_var)', 8192 * (PW // 8) * (PH // 8))
    arena = -(-top // (1 << 20)) * (1 << 20)
    for op, C, st, f, grid in steps:
        mod, sym, g, thr, ov = choose(op, A, C, grid, kdir, cache)
        if op == 'var' and mod.endswith(tsym('var', A, C) + '.co'): f = dict(f, scratch=scratch[0])
        ka = pack(structs, st, f, g)
        lines.append('%s|%s|%d|%d|%d|%d|%d%s' % (mod, sym, len(blob), len(ka), g[0], g[1], thr, '|%d' % g[2] if g[2] > 1 else ''))
        blob += ka; mods.add(mod)
        if ov: ovr[ov[0]] = '%s|%s|%s|=|=|=|%d|%d|0<-0:%d' % (ov[0], ov[1], ov[2], ov[3], ov[4], ov[4]); mods.add(ov[1])
    meta = dict(width=W, height=H, base=BASE, arena_size=arena, off_src=src, src_size=W * H * 8, off_dst=dst, dst_size=W * H * 8, skip=0,
                vit_lo=vit_lo, vit_hi=vit_hi, bn_lo=bn_lo, bn_hi=bn_hi, pre_step=next(i for i, t in enumerate(steps) if t[0] == 'pre'), post_step=next(i for i, t in enumerate(steps) if t[0] == 'post'), pad_w=PW, pad_h=PH, n_steps=len(steps), off_vitres=vitres, off_headb=headb,
                placements=[list(b) for b in bufs], **({'hybrid': '%dx%d@%d' % (WM, HM, L)} if hyb else {}))
    return dict(lines=lines, kernargs=bytes(blob), overrides=list(ovr.values()), meta=meta, wofs=wofs, modules=mods)


def translated_choice(op, A, C, grid, kdir, cache):
    mod = choose(op, A, C, grid, kdir, cache)[0]
    return mod.endswith(tsym(op, A, C) + '.co')


def pack(structs, st, f, grid):
    size, fields = structs[st]
    ka = bytearray(SLOT)
    for k, v in f.items():
        if k not in fields: raise PlanError('nr_params.h %s has no field %s' % (st, k))
        o, fmt, cnt, ptr = fields[k]
        vs = v if isinstance(v, list) else [v]
        if len(vs) > cnt: raise PlanError('%s.%s takes %d values' % (st, k, cnt))
        for j, v in enumerate(vs):
            if ptr: v = 0 if v is None else BASE + v   # pointer: arena offset -> virtual address, None = null
            elif v is None: v = 0
            struct.pack_into('<' + fmt, ka, o + j * struct.calcsize(fmt), v)
    h = (size + 7) // 8 * 8      # HIP hidden arguments
    struct.pack_into('<IIIHHH', ka, h, grid[0], grid[1], grid[2], 256, 1, 1)
    struct.pack_into('<H', ka, h + 64, 3 if grid[2] > 1 else 2)
    return bytes(ka)


# ------------------------------------------------------------------ output
def write_dir(P, out, kdir, mm, recs):
    out.mkdir(parents=True, exist_ok=True); (out / 'kernels').mkdir(exist_ok=True)
    (out / 'manifest.txt').write_text(''.join(l + '\n' for l in P['lines']))
    (out / 'kernargs.bin').write_bytes(P['kernargs'])
    (out / 'overrides.txt').write_text(''.join(l + '\n' for l in P['overrides']))
    m = P['meta']
    mid = 0xcbf29ce484222325   # model id (runtime/model_id.py, weights_ht.h nr_model_id): FNV-1a over record names + u64 sizes
    for nm, (_, n) in recs.items():
        for x in nm.encode() + b'\0' + struct.pack('<Q', n): mid = ((mid ^ x) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    m = dict(m, model_id='%016x' % mid, weights_res=RES, int8_swin=int(os.environ.get('NR_PLAN_INT8', '1')),   # int8_swin: runtime swaps in k_swin_*_i8 (env NR_INT8_SWIN overrides at load)
             vit_every=3 if RES == 'WEIGHTS_CC_SMALL_FP8' else 2)   # DLL default when nr_port.ini has no vit_every (the old model flickers more at 3)
    (out / 'meta.json').write_text(json.dumps(m, indent=1))
    (out / 'meta.txt').write_text(''.join('%s=%s\n' % (k, v) for k, v in m.items() if k != 'placements'))
    for mod in P['modules']: shutil.copy2(kdir / mod.split('\\')[-1], out / 'kernels')
    h, rows = 0xcbf29ce484222325, []
    for nm, o in sorted(P['wofs'].items(), key=lambda t: t[1]):
        fo, n = recs[nm]
        for x in mm[fo:fo + n]: h = ((h ^ x) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        rows.append('%d %s %d\n' % (o, nm, n))
    res = '' if RES == 'WEIGHTS_HT' else 'res %s\n' % RES   # runtimes without 'res' support fail cleanly on it
    (out / 'weights.map').write_text(res + 'fnv %d %d\n' % (h, len(rows)) + ''.join(rows))
    lzms(bytes(m['arena_size']), out / 'arena.lzms')


def lzms(d, path):
    import ctypes as C
    cab = C.WinDLL('cabinet'); hc = C.c_void_p()
    if not cab.CreateCompressor(5, None, C.byref(hc)): raise PlanError('CreateCompressor failed')
    o = C.create_string_buffer(1 << 20); n = C.c_size_t()
    if not cab.Compress(hc, d, C.c_size_t(len(d)), o, C.c_size_t(len(o)), C.byref(n)): raise PlanError('LZMS compress failed')
    cab.CloseCompressor(hc); Path(path).write_bytes(o.raw[:n.value])


POST_REC_BYTES = 21808   # size of block70.layer0.layer in 310.8: the record layout k_post_native_prep decodes


def fnv(b):
    h = 0xcbf29ce484222325
    for x in b: h = ((h ^ x) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def build(dll, sizes, out, kdir, use_flex, hyb=None):
    structs = parse_header()
    recs, mm, fh = dll_records(dll)
    A = derive({k: v[1] for k, v in recs.items()})
    if use_flex: flex_check(dll, A)
    fo, nb = recs['block%d.layer0.blend_scale' % A['post']]
    A['blend'] = struct.unpack('<e', mm[fo:fo + 2])[0]
    fo, nb = recs['block%d.layer0.layer' % A['post']]
    A['post_ok'] = nb == POST_REC_BYTES
    A['records'] = {k: v[1] for k, v in recs.items()}
    print('arch: levels %s | C_mid %d x%d (%d heads) | ViT %d/%d x%d | post block %d, blend %.6g%s' % (
        [(c, len(s)) for c, s in A['enc']], A['c_mid'], len(A['c1']), A['heads'], A['c_vit'], A['hid'], len(A['vit']), A['post'], A['blend'],
        '' if A['post_ok'] else ' | post record layout differs from 310.8: k_post_native not used'))
    for W, H in sizes:
        P = plan(A, W, H, kdir, structs, hyb)
        write_dir(P, Path(out) / ('%dx%d' % (W, H)), kdir, mm, recs)
        print('%dx%d: %d dispatches, arena %.1f MB, %d overrides -> %s' % (W, H, len(P['lines']), P['meta']['arena_size'] / 1e6, len(P['overrides']), Path(out) / ('%dx%d' % (W, H))))


# ------------------------------------------------------------------ selftest: synthetic shrunk weight tables
def selftest(dll):
    structs = parse_header()
    recs, mm, fh = dll_records(dll)
    base = {k: v[1] for k, v in recs.items()}
    kdir = DEFAULT_KERNELS

    def run(label, size, want):
        try:
            A = derive(size); A.update(blend=1.0, post_ok=True, records=size)
            P = plan(A, 960, 540, kdir, structs); got = 'OK %d dispatches' % len(P['lines'])
        except PlanError as e: got = 'PlanError: %s' % e
        ok = got.startswith(want)
        print('%-4s %-46s %s' % ('PASS' if ok else 'FAIL', label, got[:230])); return ok

    def renum(size, drop):   # remove block `drop`, shift later blocks down
        out = {}
        for k, v in size.items():
            b = int(re.match(r'block(\d+)', k)[1])
            if b == drop: continue
            out[re.sub(r'^block\d+', 'block%d' % (b - (b > drop)), k)] = v
        return out
    ref = {(960, 540): (576, 960), (1280, 720): (768, 1344), (1920, 1080): (1088, 1920), (2560, 1440): (1472, 2560)}   # dlssnr-pytorch-flex plan.padding
    ok = [all(geometry(w, h, 6)[:2] == v for (w, h), v in ref.items())]
    print('%-4s geometry = reference padding at 540p/720p/1080p/1440p' % ('PASS' if ok[0] else 'FAIL'))
    ok.append(run('310.8 as is', base, 'OK 156'))
    ok.append(run('drop encoder block 2 (level 0: 3 blocks)', renum(base, 2), 'PlanError: decoder level sizes'))
    s = renum(renum(base, 2), 66)    # drop a block on both sides of level 0: 3 + 3 blocks
    ok.append(run('drop a block in enc+dec level 0', s, 'OK 154'))
    s = renum(renum(base, 31), 30)   # ViT 7 blocks... (removing 30 drops the head block: must fail)
    ok.append(run('drop the head block', s, 'PlanError'))
    ok.append(run('drop a ViT block', renum(base, 31), 'OK 151'))
    ok.append(run('drop a C_mid enc + dec block', renum(renum(base, 40), 23), 'OK 148'))
    c = dict(base)                   # C_mid 512 -> 384, 12 heads (levels 24/48/96/192): no kernels at those widths
    for k in c:
        m = re.match(r'block(\d+)\.layer(\d)', k); b, l = int(m[1]), int(m[2])
        if 23 <= b <= 30 or 40 <= b <= 47: c[k] = {0: 2 * 384 * 384, 1: 384 * 384 + 768, 2: 3 * 384 * 384 + 256 * 384 + 48, 3: 384 * 384 + 768, 4: 384 * 1024 + 16}[l]
        if b == 39: c[k] = 384 * 1024 + 768
    ok.append(run('C_mid 384 / 12 heads (no kernel)', c, 'PlanError: no kernel for pre'))
    v = dict(base)                   # ViT 1024 -> 768 wide, hidden 3072
    for k in v:
        m = re.match(r'block(\d+)\.layer(\d)', k); b, l = int(m[1]), int(m[2])
        if 31 <= b <= 38: v[k] = {0: 3072 * 768 + 16, 1: 3072 * 768 + 1536, 2: 3 * 768 * 768 + 128, 3: 2, 4: 768 * 768 + 1536}[l]
        if b == 30 and l == 4: v[k] = 512 * 768 + 16
        if b == 39: v[k] = 512 * 768 + 1024
    ok.append(run('ViT 768/3072 (no kernel)', v, 'PlanError: no kernel for head'))
    print('all PASS' if all(ok) else 'SELFTEST FAILED'); return all(ok)


if __name__ == '__main__':
    a = sys.argv[1:]
    try:
        if a and a[0] == '--selftest':
            sys.exit(0 if selftest(a[1] if len(a) > 1 else r'C:/Program Files (x86)/Steam/steamapps/common/Fallout 4/nvngx_dlssnr.dll') else 1)
        if '--res' in a: RES = a[a.index('--res') + 1]
        hyb = (lambda v: (float(v[0]), int(v[1]) if len(v) > 1 else 0))(a[a.index('--hybrid') + 1].split('@')) if '--hybrid' in a else None
        kdir = Path(a[a.index('--kernels') + 1]) if '--kernels' in a else DEFAULT_KERNELS
        pos = [x for i, x in enumerate(a) if not x.startswith('--') and (i == 0 or a[i - 1] not in ('--kernels', '--res', '--hybrid'))]
        if len(pos) != 3: sys.exit(__doc__)
        build(pos[0], [tuple(int(v) for v in s.split('x')) for s in pos[1].split(',')], pos[2], kdir, '--no-flex' not in a, hyb)
    except PlanError as e:
        sys.exit('build_plan: ' + str(e))
