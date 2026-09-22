# Recovered output-address contract: SWINvar32true

Scope: pinned source image, store PC 0xc3908 only. This is static reverse
engineering checked against host traces and one corrected physical fixture,
not an authentic game capture or a complete VarParams declaration.

## Proven field use

At 0xb020c, four signed words at kernarg +0x18 are loaded into s24..s27.
At 0xb0360..0xb036c the kernel forms:

* X origin = word at +0x20 + 8 * workgroup_x (s48).
* Y origin = word at +0x24 + 8 * workgroup_y (s49).
* Height = word at +0x18 (s24); width = word at +0x1c (s25), for this path.

The legacy Python fixture calls the +0x20 argument `offy` and +0x24 `offx`.
These names are reversed relative to this address calculation. They remain
unchanged for existing callers, but the differential CLI now exposes correctly
mapped `--offset-x` and `--offset-y` and records them in evidence identifiers.

## Store formula

At 0xc383c..0xc3880, signed division by two truncates toward zero:
`h = trunc(H/2)`, `w = trunc(W/2)`, `oy = trunc(Y_origin/2)`,
`ox = trunc(X_origin/2)`. The loop covers logical IDs t=0..511 with 256 threads.

At 0xc3934..0xc3964:

```
y = oy + (t >> 7)
x = ox + ((t >> 5) & 3)
enabled = (y < h) && (x < w)
```

There is no lower-bound check in those comparisons. Assuming this path is
enabled by preceding control flow, 0xc38a0..0xc3908 computes byte displacement:

```
16 * (y * w + x) + (t & 15) + (t & 16) * h * w
```

The two channel groups therefore have 16 bytes per spatial element, row stride
`16*w`, and plane separation `16*h*w`. These are computed values for this path,
not a recovered general external stride struct.

## Explained failure and corrected experiment

With H=7, W=9 and offsets (-4,-4), h=3, w=4. Two observed writers to +0x30:

* t=48: wave 1 lane 16, first iteration; x=-1, y=-2, channel=16.
* t=416: wave 5 lane 0, second iteration; x=-1, y=1, channel=0.

Both address expressions evaluate to 48. The original-program trace confirms
the stores and different values. The GPU selected the wave-1 values for all
16 differing bytes, while the interpreter's final bytes came from wave 5.
The static audit predicts additional negative-coordinate and overlapping
addresses assuming the path is enabled; it is not a complete control-flow or
race detector and does not certify all writes in other paths.

Changing only the explicit offsets to zero in the differential fixture gives:

* Variant `<32,true>`, flags 63, seed 4, H=7 W=9, grid 1x1, 256 threads.
* 147738 emulator steps, 9462 changed reference bytes.
* Zero differing bytes across the entire arena; GPU guards intact.
* Module SHA256: c75137d2fc8a69b19e65b0b6d5e1ea0d1cc419af383ad3b7d4867349ddeaa371.
* Output SHA256: 2fa110675708e97e1b981341ee4f16bd1a7823bec4c9b902f8ba956ee3d02576.

No translation instruction or tolerance changed for this result. The old
negative-origin failure remains recorded. Default offsets are not silently
changed, so old tests retain their original meaning.

Reproduce inside the existing bounded sandbox:

```
python rdna2/emu/difftest_var.py 32_1 63 4 1 1 --height 7 --width 9 --offset-x 0 --offset-y 0
```

Host-only address audit:

```
python rdna2/emu/var_output_contract.py 7 9
python rdna2/emu/var_output_contract.py 7 9 --offset-x 0 --offset-y 0
```

## Remaining runtime evidence

Do not infer that negative origins are forbidden in every authentic launch:
the caller may use shifted pointers, padding or a different mode. No supplied
host declaration has resolved the full VarParams layout. In particular the
inferred pointer fields +0x30..+0x68, +0x88 and +0x90 require classification.
Capture explicit argument bytes, allocation bases/sizes, any pointer-to-view
offsets, dimensions, launch grid, flags and the actual dispatch sequence.
Then validate each output path's address ranges and ownership against those
allocations. A zero-origin synthetic pass does not resolve those contracts,
provide real weights, or demonstrate game rendering.
