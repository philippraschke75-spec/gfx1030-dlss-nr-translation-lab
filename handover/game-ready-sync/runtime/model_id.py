# Model id of an nvngx_dlssnr.dll (same as nr_model_id in weights_ht.h): FNV-1a 64 over every WEIGHTS_HT record's
# name, a 0 byte and its u64 size. 310.8 (original and ShortFuse) = 1ff9997855c80dd5.
# With data dirs: writes "model_id" into each <dir>/meta.json; the DLL turns NR off when it differs from the user's DLL.
# Multi-model layout: nr_data\<model_id>\<W>x<H>\ (flat nr_data\<W>x<H>\ = whatever its meta.json says, default 310.8).
# usage: python model_id.py <nvngx_dlssnr.dll> [<data dir with meta.json> ...]
import sys, os, mmap, struct, json
sys.path.insert(0, r'E:/dlss 5/rdna2/emu'); import weights_ht


def model_id(path):
    f = open(path, 'rb'); m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ); h = 0xcbf29ce484222325
    for r in weights_ht.records(m):
        for x in r['name'].encode() + b'\0' + struct.pack('<Q', r['bytes']): h = ((h ^ x) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return '%016x' % h


def stamp(dll, d):
    p = os.path.join(d, 'meta.json'); j = json.load(open(p)); j['model_id'] = model_id(dll); json.dump(j, open(p, 'w'), indent=1)


if __name__ == '__main__':
    print(model_id(sys.argv[1]))
    for d in sys.argv[2:]: stamp(sys.argv[1], d)
