"""Host-only check for order-sensitive synthetic SWIN fixtures."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import gfx11emu as E
import run_var as V
import run_emu as R

def main():
    flags = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    V.ARENA = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0x404010000
    prog = E.load_program(R.DIS, {'_Z10k_swin_varILi32ELb1EEv9VarParams'})
    order = [(0, 0), (1, 0), (0, 1), (1, 1)]
    outputs = []
    for sequence in (order, list(reversed(order))):
        g, ka = V.build(1, V.make_kernarg(flags=flags, grid=(2, 2)), len(V.PTR_FIELDS))
        for x, y in sequence:
            E.run_workgroup(prog, g, 15616, 256,
                            {0: ka & 0xffffffff, 1: ka >> 32, 14: x, 15: y},
                            max_steps=8_000_000)
        outputs.append(g.regions[1].arr.copy())
    diff = np.flatnonzero(outputs[0] != outputs[1])
    result = dict(flags=flags, device_base=hex(V.ARENA),
                  order_sensitive_bytes=len(diff),
                  first_offsets=[hex(int(i)) for i in diff[:16]],
                  hashes=[hashlib.sha256(o.tobytes()).hexdigest() for o in outputs])
    print(json.dumps(result, indent=2))
    Path(__file__).resolve().parent.parent.joinpath('build', 'workgroup_order_%d.json' % flags).write_text(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
