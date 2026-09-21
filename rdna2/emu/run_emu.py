import sys, time, struct, json
import numpy as np
sys.path.insert(0, '.')
import gfx11emu as E
DIS = '../../analysis/gfx1100-disassembly.txt'
KERNEL = '_Z16k_swin_1h_32_fp810SwinParams'
HELPER = '_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti'

def build(seed=1, H=16, W=16, offy=-4, offx=-4, p40=256, blob_bytes=1 << 22, io_bytes=1 << 22):
    rng = np.random.default_rng(seed)
    g = E.GMem()
    KA, DP, IN, OUT, BLOB = 0x7000_0000_0000, 0x7100_0000_0000, 0x7200_0000_0000, 0x7300_0000_0000, 0x7400_0000_0000
    ka = bytearray(296)
    struct.pack_into('<QQQ', ka, 0, IN, OUT, BLOB)
    struct.pack_into('<iiii', ka, 24, H, W, offy, offx)
    struct.pack_into('<I', ka, 40, p40)
    struct.pack_into('<I', ka, 52, 256)
    dp = bytearray(64); struct.pack_into('<HHHH', dp, 4, 256, 1, 1, 0)
    g.add('kernarg', KA, ka); g.add('dispatch', DP, dp)
    g.add('in', IN, rng.integers(0, 256, io_bytes, dtype=np.uint8))
    g.add('out', OUT, np.full(io_bytes, 0xA5, np.uint8))
    g.add('blob', BLOB, rng.integers(0, 256, blob_bytes, dtype=np.uint8))
    return g, dict(KA=KA, DP=DP)

if __name__ == '__main__':
    g, a = build()
    prog = E.load_program(DIS, {KERNEL, HELPER}); print('instructions', len(prog))
    # entry = first instruction of the kernel (lowest address is the helper: find kernel start)
    kstart = min(i for i, p in enumerate(prog) if p[0] == 0x2cd00)
    # reorder so the kernel entry is index 0 is unnecessary: set pc explicitly
    ex_prog = prog
    sg = {0: a['DP'] & 0xffffffff, 1: a['DP'] >> 32, 2: a['KA'] & 0xffffffff, 3: a['KA'] >> 32, 14: 0, 15: 0}
    t = time.time()
    # patch: run_workgroup starts waves at pc 0; rotate program so kernel is first
    ex_prog = prog[kstart:] + prog[:kstart]
    r = E.run_workgroup(ex_prog, g, 62592, 256, sg)
    print('steps', r['steps'], 'time %.1fs' % (time.time() - t))
    for reg in g.regions: print(reg.name, 'rd' if reg.rd else '', 'wr' if reg.wr else '', hex(reg.lo) if reg.hi else '-', hex(reg.hi))
