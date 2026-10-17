"""Run our network (nr_runtime, GPU) or the PyTorch reference (CPU) on a set of frames x conditioning settings.

    py -I tools/iq_run.py ours <nr_runtime dir> <nr_data/WxH> <W> <H> <in.npy> <out.npy> <settings>      (wrap in gpu2.sh)
    <ref venv python> tools/iq_run.py ref <nvngx_dlssnr.dll> <W> <H> <in.npy> <out.npy> <settings> [res]

in.npy: float32 [N, H, W, 3] proxy frames in [0,1]. out.npy: float32 [S, N, H, W, 3], the network output at intensity 1
(clamp(frame + residual/4)), single frame (no history). settings: ';'-separated "tone,structure,skin,auto_mask"
(e.g. "1,1,-1,1;1.5,1.5,-1,1"), optional 5th field style 0..2. res: WEIGHTS_CC_SMALL_FP8 (new model) or WEIGHTS_HT (old).
"""
import os, sys
import numpy as np


def parse(s):
    return [tuple(float(v) for v in (t + ',0').split(',')[:5]) for t in s.split(';')]   # style defaults to 0


def ours(dlldir, data, W, H, inp, out, settings):
    import ctypes as C
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from iq_lib import to_half4
    os.add_dll_directory(dlldir); lib = C.CDLL(os.path.join(dlldir, 'nr_runtime.dll'))
    lib.nr_create.restype = C.c_void_p; lib.nr_create.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_int]
    lib.nr_run2.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
    lib.nr_set_conditioning.argtypes = [C.c_void_p, C.c_int, C.c_float, C.c_float, C.c_float, C.c_int]
    err = C.create_string_buffer(512); c = lib.nr_create(data.encode(), W, H, 1, err, 512); assert c, err.value
    frames = np.load(inp); res = np.zeros((len(settings),) + frames.shape, np.float32); ms = C.c_float()
    for si, (t, s, k, m, sty) in enumerate(settings):
        assert lib.nr_set_conditioning(c, int(sty), t, s, k, int(m)) == 0, (t, s, k, m, sty)
        for n, f in enumerate(frames):
            x = to_half4(f); y = np.empty_like(x)
            assert lib.nr_run2(c, x.ctypes.data, y.ctypes.data, C.byref(ms), 0) == 0
            res[si, n] = y.view(np.float16)[..., :3].astype(np.float32)
    print('ours %dx%d: %d settings x %d frames, last %.2f ms' % (W, H, len(settings), len(frames), ms.value))
    np.save(out, np.clip(np.nan_to_num(res), 0, 1))


def ref(dll, W, H, inp, out, settings, wres='WEIGHTS_CC_SMALL_FP8'):
    import torch, contextlib, io
    import dlssnr.extract as ex; ex.RESOURCE_NAME = wres
    from dlssnr.extract import extract
    from dlssnr.io import Controls, _mirror
    from dlssnr.model import NRNet, _main
    with contextlib.redirect_stdout(io.StringIO()): w = extract(dll, None)
    net = NRNet(w, W, H)
    rows, cols = _mirror(torch.arange(net.padding.height), H), _mirror(torch.arange(net.padding.width), W)
    pad = lambda img: torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[:, rows][:, :, cols].unsqueeze(0).contiguous()
    frames = np.load(inp); res = np.zeros((len(settings),) + frames.shape, np.float32)
    with torch.no_grad():
        for si, (t, s, k, m, sty) in enumerate(settings):
            ctl = Controls(seed=0, style=int(sty), local_tone=t, local_structure=s, skin_structure=k, auto_mask=bool(m))
            for n, f in enumerate(frames):
                res[si, n] = _main(net.run(pad(f), ctl)[len(net.plan) - 1])[0, :, :H, :W].clamp(0, 1).permute(1, 2, 0).numpy()
    print('ref %dx%d done' % (W, H))
    np.save(out, res)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['ours']: ours(a[1], a[2], int(a[3]), int(a[4]), a[5], a[6], parse(a[7]))
    elif a[:1] == ['ref']: ref(a[1], int(a[2]), int(a[3]), a[4], a[5], parse(a[6]), *a[7:8])
    else: sys.exit(__doc__)
