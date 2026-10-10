"""prof.py <dll dir> <data dir> <out.npy> [frames]: e2e.py under NR_PROF=1; prints the per-kernel sums (ms) of the
C=512 kernels for frame 5 plus the frame time. Run under the GPU lock."""
import os, re, subprocess, sys
d, data, outp = sys.argv[1], sys.argv[2], sys.argv[3]
frames = sys.argv[4] if len(sys.argv) > 4 else "8"
e2e = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "handover", "game-ready-sync", "native", "e2e.py")
env = dict(os.environ, NR_PROF="1")
r = subprocess.run([sys.executable, e2e, d, data, outp, frames], capture_output=True, text=True, env=env, timeout=140)
tot = {}
for m in re.finditer(r"prof\s+\d+\s+([\d.]+)\s+(\S+)", r.stderr):
    tot[m.group(2)] = tot.get(m.group(2), 0.0) + float(m.group(1))
print(r.stdout.strip())
print("  ".join(f"{k} {tot.get(k, 0):.3f}" for k in ("k_ffwd2_n", "k_conv2_n", "k_qkv_attn_n")), f" sum {sum(tot.get(k, 0) for k in ('k_ffwd2_n', 'k_conv2_n', 'k_qkv_attn_n')):.3f}  frame {sum(tot.values()):.2f}")
if r.returncode: print("FAILED", r.returncode, r.stderr[-500:])
