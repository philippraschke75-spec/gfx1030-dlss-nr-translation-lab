"""How large are GPU-vs-emulator differences, measured in e4m3 code steps (0 = identical, 1 = adjacent code)?"""
import sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import difftest_var as D, run_var as V
key, flags = sys.argv[1], int(sys.argv[2])
sym, lds = D.SYMS[key]; lds = lds or D.group_size(sym)
ka, init, ref, steps, secs = D.emulate(sym, lds, flags, 1, (2, 2))
tag = 'v%s_f%d_s1' % (key, flags)
out = np.fromfile(D.OUT / tag / 'arena.bin', np.uint8)
diff = np.nonzero(out != ref)[0]
ordn = lambda b: np.where(b & 0x80, -(b & 0x7f).astype(np.int16), (b & 0x7f).astype(np.int16))
d = np.abs(ordn(ref[diff]) - ordn(out[diff]))
print('%s flags=%d: %d mismatching bytes of %d written' % (key, flags, len(diff), int((ref != init).sum())))
print('code-step distribution:', dict(sorted(collections.Counter(np.minimum(d, 8).tolist()).items())), '(8 = 8 or more)')
by = collections.Counter(int(i // V.SLOT) for i in diff)
print('by parameter slot:', {hex(V.PTR_FIELDS[k]): v for k, v in by.items()})
