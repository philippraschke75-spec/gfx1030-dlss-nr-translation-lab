# Makes a self-contained runtime data dir from a gen_runtime_data.py output: hardlinks the big files, snapshots the
# kernel .co files the manifest names (so later kernel rebuilds in the repo cannot desync them) and rewrites the paths.
# usage: python make_data_dir.py <src data dir> <dst data dir> <kernel dir>
import os, shutil, sys
src, dst, kdir = sys.argv[1:4]
os.makedirs(os.path.join(dst, "kernels"), exist_ok=True)
for f in ("arena.bin", "kernargs.bin", "meta.txt", "meta.json"):
    d = os.path.join(dst, f)
    if not os.path.exists(d): os.link(os.path.join(src, f), d)
out = []
for line in open(os.path.join(src, "manifest.txt")):
    t = line.rstrip("\n").split("|")
    name = t[0].replace("/", "\\").rsplit("\\", 1)[-1]
    k = os.path.join(dst, "kernels", name)
    if not os.path.exists(k): shutil.copy2(os.path.join(kdir, name), k)
    t[0] = os.path.abspath(k)
    out.append("|".join(t))
open(os.path.join(dst, "manifest.txt"), "w").write("\n".join(out) + "\n")
print(len(out), "dispatches,", len(os.listdir(os.path.join(dst, "kernels"))), "kernels ->", dst)
