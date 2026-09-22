"""Host-only: verify k_import format=0 decodes RGBA16F correctly (R,G from one 4B dword, B from a 2B load, per
the disassembly at 0xAA440-0xAA488: global_load_b32 (R|G<<16) + global_load_u16 (B), three v_cvt_f32_f16).
Feeds known half-float pixel values, checks the network-input (RGB32F) destination reproduces them (mode 0 = raw decode)."""
import struct
import numpy as np
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z8k_import12ImportParams'
H, W, PH, PW = 4, 4, 8, 8       # padded >= source so every source pixel is in-bounds
known = np.array([0.125, 2.5, 65.0, 0.0009, 1.0, -3.0], dtype=np.float16)   # representative HDR-ish values incl. small/large/negative(clamped?)
pitch = W * 8   # 8 bytes/pixel = RGBA16F row pitch
grid = ((PW + 255) // 256, PH)
ka = bytearray(0x130)
struct.pack_into('<Q', ka, 0x00, V.ARENA)
struct.pack_into('<iiiiii', ka, 0x08, pitch, 0, H, W, PH, PW)   # format = 0
struct.pack_into('<Q', ka, 0x20, V.ARENA + V.SLOT)
struct.pack_into('<if', ka, 0x28, 0, 1.0)                        # mode = 0 (raw decode), scale = 1.0
struct.pack_into('<III', ka, 0x30, grid[0], grid[1], 1)
struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)
prog = E.load_program(R.DIS, {SYM})
g, KA = V.build(1, ka, len(V.PTR_FIELDS))
a = g.regions[1].arr
# pixel 0 (row0,col0): R,G,B = known[0..2]; pixel 1 (row0,col1): R,G,B = known[3..5]
px0 = struct.pack('<4e', known[0], known[1], known[2], np.float16(1.0))
px1 = struct.pack('<4e', known[3], known[4], known[5], np.float16(1.0))
a[0:8] = np.frombuffer(px0, np.uint8); a[8:16] = np.frombuffer(px1, np.uint8)
lds = D.group_size(SYM)
for wy in range(grid[1]):
    for wx in range(grid[0]):
        E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=2_000_000)
dst = a[V.SLOT:V.SLOT + PH * PW * 12].view('<f4').reshape(PH, PW, 3)
print('pixel(0,0) decoded RGB:', dst[0, 0], ' expected:', known[0:3].astype(np.float32))
print('pixel(0,1) decoded RGB:', dst[0, 1], ' expected:', known[3:6].astype(np.float32))
print('match pixel0:', np.allclose(dst[0, 0], known[0:3].astype(np.float32), rtol=1e-3))
print('match pixel1:', np.allclose(dst[0, 1], known[3:6].astype(np.float32), rtol=1e-3))
