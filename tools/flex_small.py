"""Run dlssnr-pytorch-flex on the 310.9 small model (WEIGHTS_CC_SMALL_FP8) without editing flex.

    <flex venv python> tools/flex_small.py <nvngx_dlssnr.dll> [WEIGHTS_CC_SMALL_FP8] -- <dlssnr CLI args>
e.g. ... tools/flex_small.py new.dll -- run frame.png -o out.png --device cpu
Without CLI args it prints the architecture flex derives. Needs PYTHONPATH=<flex dir>, DLSSNR_TRITON=0.
Same container framing as WEIGHTS_HT (u64 size, records), so only the resource name differs.
"""
import sys
import dlssnr.extract as ex

a = sys.argv[1:]; cli = a[a.index('--') + 1:] if '--' in a else []; a = a[:a.index('--')] if '--' in a else a
dll = a[0]; ex.RESOURCE_NAME = a[1] if len(a) > 1 else 'WEIGHTS_CC_SMALL_FP8'
if not cli:
    from dlssnr.plan import Arch
    w = ex.extract(dll, None); A = Arch.from_weights(w)
    print('levels', A.levels, 'split', A.split_channels, 'x', A.split_blocks, 'vit', A.vit_channels, A.vit_hidden, 'x', A.vit_blocks)
else:
    import runpy
    sys.argv = ['dlssnr'] + cli + ['--dll', dll]
    runpy.run_module('dlssnr', run_name='__main__')
