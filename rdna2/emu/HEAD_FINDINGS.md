# k_final_head: output address and the right grid

## Output address (gfx1100 disassembly, `0xa3c00..`)

```
a3c38 s_lshl_b64 s[12:13], s[2:3], 13    ; s[2:3] = sext(s15): wg_id_y * 8192  -> INPUT pointer (s4)
a42a4 s_lshl_b64 s[0:1], s[2:3], 14      ; wg_id_y * 16384
a42b0 s_add_u32  s2, s6, s0              ; s[2:3] = out + wg_id_y * 16384      -> OUTPUT pointer
a4c04 global_store_b8 v12, v11, s[2:3]   ; the output store
```

The output pointer IS offset per workgroup, only later than the input pointer. `s14` (workgroup_id_x) is never read.
So the kernel is 1D: one workgroup covers 8192 B of input and 16384 B of output (16330 B written).

## Why every grid tried wrote one workgroup's share

The **translated** gfx1030 kernel enables `workgroup_id_x` only (`.amdhsa_system_sgpr_workgroup_id_y 0`) and its
prologue is `s_mov_b32 s15, s2`. The original's "workgroup_id_y" therefore arrives in the hardware **X** id. A grid in
Y gives every workgroup the same id, so all 960/1601/12810 y-workgroups wrote the same 16330 B. That is why
`(7,960)` gave `7 x 16330` (7 distinct X ids) and `(1,12810)` gave `1 x 16330`.

## Grid

`gx = ceil(W*H*16 / 16384)`, `gy = 1`: **(1601, 1)** at 1707x960 (26 219 520 B / 16384 = 1600.3).

## Measured (1707x960, head buffer = 26 219 520 B)

| head input | grid | nonzero | distinct bytes |
|---|---|---|---|
| decoder chain (default) | (7,960) | 0.437% | 2 |
| decoder chain | (1,1601) | 0.062% | 2 |
| enc s1 pool (varying, `HEAD_SRC=enc1`) | (1,1601) | 0.062% | 173 |
| enc s1 pool | (7,960) | 0.437% | 220 |
| **enc s1 pool** | **(1601,1)** | **99.999%** | **255** |
| decoder chain | (1601,1) | 100% | **1** |

The row 3-from-bottom result confirms the grid derivation with real varying data.

## Success criterion NOT met on the real chain

With the real chain the head is now fully written but constant, because its **input is constant**
(`stage_in` = last decoder block: 1601 chunks, all nonzero, **1 distinct byte**). The collapse is upstream of the head:

* encoder pools are healthy (values up to +-448, 24.5-25% nonzero)
* `c512_1 w0..3`: 0.13-0.49% nonzero, 2 distinct
* `vit b0..5`: 0.01-0.03% nonzero, 2 distinct; `block39 out`: 0% (1 distinct)
* every decoder ping/pong/pool: 2 distinct bytes

So the first bad buffers are the 512-channel blocks (23-30). `c512_stage`, the ViT blocks and `k_dec_upsample` all
launch at grid (1,1) at 60x106 geometry. That is the same class of bug (one workgroup) but I did **not** test a fix
there and did not check whether those kernels also lost workgroup_id_y in translation. Next step.

## Changes (net_frame_full.py)

`HEAD_GRID` gains `out16k` (1,N) and `x16k` (N,1, now the default); `HEAD_SRC=enc1` isolation option; reports for head
input/output, mid-network and decoder buffers (nonzero % and distinct bytes).
