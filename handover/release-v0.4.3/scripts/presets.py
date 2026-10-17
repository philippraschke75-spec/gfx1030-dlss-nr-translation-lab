"""presets.py: writes E:/dlss5-v0.4.3/nr_port.ini (= Standard) and presets/*.ini, each a full nr_port.ini (v0.4.2 base + key edits)."""
import re
P = r'E:/dlss5-v0.4.3'
base = open(r'E:/dlss5-v0.4.2/nr_port.ini', encoding='utf-8').read().replace('v0.4.2', 'v0.4.3')
base = base.replace('; Model resolution: 960x540 (recommended), 1280x720 (slower); 1920x1080 or 2560x1440 for image-quality comparison (much slower)',
                    '; Model resolution: 960x540 (recommended), 1280x720 (better faces, slower); 1920x1080 or 2560x1440 (most detail, much slower)')
assert base.count('\nhistory=1\n') == 1
base = base.replace('\nhistory=1\n', '\nhistory=1\n; Edit stabilisation: blends the shown NR edit with the previous one (moved by the game\'s motion vectors).\n'
                    '; Removes most of the remaining flicker at no cost. 0 = off (the v0.4.2 image), up to 0.95.\nedit_temporal=0.7\n')
base = base.rstrip('\n') + '\n' +'''; Intensity above 1 keeps the hue (NVIDIA's rule: brightness ratio ^ intensity, at most 2x either way).
; compose: add (default) | nv (NVIDIA's blend, for setups copied from NVIDIA users) | ratio (OkLab, janblade's Transfer 2)
; detail: live-frame detail above the model resolution (1 = as is)
'''

def mk(edits=(), extra=()):
    t = base
    for k, v in edits:
        t, n = re.subn(rf'(?m)^{k}=.*$', f'{k}={v}', t); assert n == 1, k
    return t + ''.join(f'{k}={v}\n' for k, v in extra)

STRONG = (('local_tone', 2), ('intensity', 1.3))
presets = {
 'Standard': ("Standard (default): NVIDIA's default look, 960x540 model.",
              "Cost: the baseline (about 43 fps in Fallout 4 at 1440p on an RX 6900 XT).",
              "Use: start here.", (), ()),
 'Strong': ("Strong: clearly stronger look, still clean: local tone 2, intensity 1.3 (+45 % edit, shadows +41 %, faces +35-85 %).",
            "Cost: no extra GPU time; a little more flicker.",
            "Use: when Standard looks too subtle. These keys override the OptiScaler NR sliders (Pass 1).", (), STRONG),
 'Cinematic': ("Cinematic: Strong + NVIDIA's Cinematic style (about 2.5x the default edit, warmer graded colour, brighter shadows).",
               "Cost: no extra GPU time; more flicker. Most realistic with a higher model resolution (model_res=1280x720 or more).",
               "Use: screenshots, slow games, the strongest NVIDIA look.", (), STRONG + (('style', 2),)),
 'HighQuality_720p': ("High quality: 1280x720 model. Faces and skin follow the full-resolution result much better.",
                      "Cost: network about +55 % vs 960x540 (see README for fps).",
                      "Use: when you have fps to spare; combine with Strong/Cinematic keys for the most realistic look.", (('model_res', '1280x720'),), ()),
 'HiRes_1080p': ("HiRes 1080p: 1920x1080 model, finer network detail.",
                 "Cost: network about 3x of 960x540; not real-time in heavy games.",
                 "Use: image-quality comparison, screenshots.", (('model_res', '1920x1080'),), ()),
 'Native_1440p': ("Native 1440p: the network at full 1440p (new model only).",
                  "Cost: network about 5x of 960x540, slowest.",
                  "Use: image-quality comparison, screenshots.", (('model_res', '2560x1440'),), ()),
 'Performance': ("Performance: 640x360 model, middle stage every 4th job, network over 4 frames (softer).",
                 "Cost: the cheapest setting.",
                 "Use: weaker GPUs or high-fps games.", (('model_res', '640x360'), ('vit_every', 4), ('nr_split', 4)), ()),
 'Community': ("Community: for users copying NVIDIA multipass setups (OptiScaler 2 passes, intensity 2, tone/structure 2):",
               "compose=nv (NVIDIA's own blend and intensity rule) and pass2_gain=1 (second pass at full strength, as on NVIDIA). Sliders follow OptiScaler.",
               "Cost: no extra per pass; each OptiScaler pass runs the network once. Strong host settings flicker noticeably more.", (('pass2_gain', 1),), (('compose', 'nv'),)),
}
for name, (a, b, c, ed, ex) in presets.items():
    hdr = f'; Preset {name} (DLSS 5 RDNA2 port v0.4.3). Copy this file over nr_port.ini (next to the game exe), then restart the game.\n; {a}\n; {b}\n; {c}\n'
    txt = hdr + mk(ed, ex)
    open(f'{P}/presets/{name}.ini', 'w', encoding='utf-8', newline='\r\n').write(txt)
    if name == 'Standard': open(f'{P}/nr_port.ini', 'w', encoding='utf-8', newline='\r\n').write(mk())
print('ok', list(presets))
