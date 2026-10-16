"""Our plan's output vs the MIT PyTorch reference (dlssnr-pytorch-flex) on one test frame.

    py -I tools/torch_ref.py ours <nr_runtime.dll dir> <nr_data/WxH> <W> <H> <out.npy>     (GPU: wrap in gpu2.sh)
    <flex venv python> tools/torch_ref.py torch <nvngx_dlssnr.dll> <W> <H> <ours.npy>        (CPU, from the flex folder)

Input: E:/dlss-runtime/color_WxH.bin with RGB clamped to [0,1] (the data runs import/export mode 0, so both sides get
the same linear values; as handover/game-ready-sync/temporal/flicker_torch.py). Prints PSNR (peak 1) ours vs PyTorch.
"""
import os, sys
import numpy as np
HIST = int(os.environ.get('HIST', '0'))   # 1: compare frame 1 of a static history sequence
BLEND = 0.7397


def frame(W, H):
    x = np.fromfile('E:/dlss-runtime/color_%dx%d.bin' % (W, H), np.uint16).reshape(H, W, 4).view(np.float16).astype(np.float32)
    x[..., :3] = np.clip(np.nan_to_num(x[..., :3]), 0, 1)
    return x


def ours(dlldir, data, W, H, out):
    import ctypes as C, os
    os.add_dll_directory(dlldir); lib = C.CDLL(os.path.join(dlldir, 'nr_runtime.dll'))
    lib.nr_create.restype = C.c_void_p; lib.nr_create.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_int]
    lib.nr_run2.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
    err = C.create_string_buffer(512); c = lib.nr_create(data.encode(), W, H, 1, err, 512); assert c, err.value
    x = np.ascontiguousarray(frame(W, H).astype(np.float16).view(np.uint16)); y = np.empty_like(x)
    for k in range(1 + HIST):   # HIST: a static 2-frame sequence, frame 1 with the runtime's history (= frame 0's output)
        assert lib.nr_run2(c, x.ctypes.data, y.ctypes.data, None, int(k > 0)) == 0
    np.save(out, y.view(np.float16)[..., :3].astype(np.float32))


def torch_ref(dll, W, H, ours_npy):
    import torch, contextlib, io
    from dlssnr.extract import extract
    from dlssnr.io import Controls, _mirror
    from dlssnr.model import NRNet, _main
    with contextlib.redirect_stdout(io.StringIO()): w = extract(dll, None)
    net = NRNet(w, W, H)
    ci = frame(W, H)[..., :3]
    t = torch.from_numpy(np.ascontiguousarray(ci.transpose(2, 0, 1)))
    rows, cols = _mirror(torch.arange(net.padding.height), H), _mirror(torch.arange(net.padding.width), W)
    pad = lambda img: torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[:, rows][:, :, cols].unsqueeze(0).contiguous()
    run = lambda: _main(net.run(pad(ci), Controls(seed=0))[len(net.plan) - 1])[0, :, :H, :W].clamp(0, 1).permute(1, 2, 0).numpy()
    with torch.no_grad():
        ref = run()
        if HIST:   # frame 1 with history = frame 0's output, as flicker_torch.py 'hist' (pre lanes 7-9 + post lerp), mirrored padding
            from dlssnr.layers import pre_post
            hp = pad(ref); feats = pre_post.PreBlock.features
            pl = next(l for l in net.layers if isinstance(l, pre_post.PostBlock)); s = pl.spec
            head_a = w.f16(s.block, s.layer, 'out_project', 16, pre_post.CHANNELS)[3]

            def features(self, image):
                f = feats(self, image).clone(); f[:, 7:10] = pre_post._centred_colour(hp); return f

            def post_forward(self, x, skip, fr):
                wide = pre_post.swin_core(self.params, self.blend(x, skip), self.spec.shift_x, self.spec.shift_y)
                res = pre_post.compose(fr, torch.einsum('rc,bchw->brhw', self.project, wide), self.controls.intensity)
                a = torch.einsum('c,bchw->bhw', head_a, wide)[:, None]
                return res + (BLEND * torch.sigmoid(a)).clamp(0, 1) * (hp - res)
            pre_post.PreBlock.features, pre_post.PostBlock.forward = features, post_forward
            ref = run()
    o = np.clip(np.load(ours_npy), 0, 1)
    psnr = lambda a, b: 10 * np.log10(1 / max(float(((a - b) ** 2).mean()), 1e-12))
    print('%dx%d: ours vs PyTorch %.2f dB (input vs PyTorch %.2f dB), edit correlation %.3f' % (
        W, H, psnr(o, ref), psnr(ci, ref), np.corrcoef((o - ci).ravel(), (ref - ci).ravel())[0, 1]))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['ours']: ours(a[1], a[2], int(a[3]), int(a[4]), a[5])
    elif a[:1] == ['torch']: torch_ref(a[1], int(a[2]), int(a[3]), a[4])
    else: sys.exit(__doc__)
