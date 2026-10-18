#!/bin/bash
D=/e/dlss-runtime/look_build; mkdir -p "$D/presets"; R="E:/dlss 5/.claude/worktrees/agent-af28cbe7506173602/handover/release-v0.4.3/nr_port.ini"
I8=${I8:-1}
tr -d '\r' < "$R" | sed -e 's/v0\.4\.3\./look-rc test build (v0.4.3 + temporal-audit + cond-audit + effect-strength, linear intensity)./' \
  -e 's/^model_res=960x540/; model_res= (keep your own value, e.g. 960x540)/' -e "s/^int8_swin=1/int8_swin=$I8/" \
  -e 's/^; Intensity above 1 keeps the hue.*/; Intensity above 1: linear blend proxy + I*(styled - proxy) (OpenDLSS-NR), clamped. intensity_rule=ratio = old brightness-ratio rule/' > "$D/nr_port.ini"
cat >> "$D/nr_port.ini" <<'EOF'
intensity_rule=linear
; NVIDIA history rules (temporal-audit): no colour gate, Catmull-Rom history warp, frame-counter seed. 1 / bilinear / 0 = v0.4.3
history_gate=0
history_filter=catmull
history_seed=frame
; history padding: reflect (NVIDIA) | zero (v0.4.3)
history_pad=reflect
; dump_frame=N: write frame N's inputs/output next to the log (debug), 0 = off
dump_frame=0
EOF
mk(){ sed -e "s/^; model_res=.*/model_res=$1/" -e '/^nr_split=/d' -e '/^edit_temporal=/d' -- "$D/nr_port.ini" > "$D/presets/Showcase_$2.ini"
  printf 'nr_split=0\nedit_temporal=0.5\nstyle=2\nintensity=2\nlocal_tone=2\nlocal_structure=1.88\nauto_mask=1\n' >> "$D/presets/Showcase_$2.ini"; }
mk 1920x1080 1080p; mk 960x540 540p
cp "$D/build/nr_runtime.dll" "$D/build/nvngx.dll_dlssnr.dll" "$D/"
