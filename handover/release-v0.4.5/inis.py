# v0.4.5 package inis: nr_port.ini + presets (run once on E:\dlss5-v0.4.5)
import glob, re
P = 'E:/dlss5-v0.4.5'
SPLIT = ('; Split the network over N frames (2, 3 or 4; 0 = off). v0.4.5: 0 by default. nr_split=3 gives more fps but the\n'
         '; shown edit may visibly trail in game (a frame or two behind, under investigation).\n')
NEW = '''; v0.4.5 look: intensity above 1 = linear blend proxy + I*(styled - proxy) (OpenDLSS-NR), clamped; ratio = the old brightness-ratio rule
intensity_rule=linear
; NVIDIA history rules: no colour gate, Catmull-Rom history warp, reflected padding. 1 / bilinear / zero = v0.4.4
history_gate=0
history_filter=catmull
history_pad=reflect
; history_seed: 0 (default) | frame. A per-frame seed (NVIDIA's rule) caused white noise in game, keep 0
history_seed=0
; Debug: dump_frame=N writes frame N's inputs/output next to the log (dump_frames = how many consecutive frames, up to 8); 0 = off
dump_frame=0
dump_frames=1
; Motion vector scale (1 = as the game sends them); the log prints a "motion:" line once
mv_scale=1
'''
NOTE = '; v0.4.5: nr_split=0 and history_seed=0 in every preset (nr_split=3 is faster but may trail in game; a per-frame seed caused white noise).\n'

def fix(s, main):
    s = s.replace('\r\n', '\n').replace('v0.4.4', 'v0.4.5').replace('v0.4.3', 'v0.4.5')
    s = re.sub(r'; Split the network over N frames.*\n; a frame or two older.*\n', SPLIT, s)
    s = re.sub(r'(?m)^nr_split=\d+$', 'nr_split=0', s)
    s = re.sub(r'(?m)^history_seed=.*$', 'history_seed=0', s)
    s = s.replace("; Intensity above 1 keeps the hue (NVIDIA's rule: brightness ratio ^ intensity, at most 2x either way).\n", '')
    if 'intensity_rule=' not in s: s = s.rstrip('\n') + '\n' + NEW
    for k, v in (('dump_frames', '1'), ('mv_scale', '1')):
        if not re.search(r'(?m)^%s=' % k, s): s += '%s=%s\n' % (k, v)
    if main:
        s = re.sub(r'(?m)^model_res=.*$', 'model_res=1280x720', s)
        s = s.replace('960x540 (recommended), 1280x720 (better faces, slower)', '1280x720 (default, better faces), 960x540 (faster)')
        s = re.sub(r'(?m)^edit_temporal=.*$', 'edit_temporal=0.3', s)
    else:
        s = re.sub(r'(?m)^(\[nr\])$', NOTE.rstrip('\n') + r'\n\1', s, count=1)
    assert re.search(r'(?m)^nr_split=0$', s) and re.search(r'(?m)^history_seed=0$', s), s[:200]
    return s.replace('\n', '\r\n')

for f in [P + '/nr_port.ini'] + glob.glob(P + '/presets/*.ini'):
    s = open(f, encoding='utf-8', newline='').read()
    open(f, 'w', encoding='utf-8', newline='').write(fix(s, f.endswith('nr_port.ini')))
    print('ok', f)
