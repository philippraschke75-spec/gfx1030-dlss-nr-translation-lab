"""Repair only the proven descriptor gap in the pinned diagnostic runtime copy."""
from pathlib import Path
import hashlib
import json
import struct

root = Path(__file__).resolve().parent.parent
source = root/'gfx1032-dlss-nr-research-main/version.dll'
backup = source.with_name('version.before-bundle-header-repair.dll')
if backup.exists():
    source = backup
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'fa616204dd68521217875fb41835d3189c89e7c87c3050e069df16bafc458a05'
start = 0x9b600
assert raw[start:start+24] == b'__CLANG_OFFLOAD_BUNDLE__'
assert struct.unpack_from('<Q', raw, start+24)[0] == 9
assert raw[start+142:start+150] == bytes(8)
bundle = bytearray(raw[start:])
# Recover seven remaining descriptors after the exact eight-byte gap.
pos = 150
for _ in range(7):
    offset, size, length = struct.unpack_from('<QQQ', bundle, pos)
    assert 0 < length < 256 and offset >= 4096 and offset+size <= len(bundle)
    assert bundle[pos+24:pos+24+length].startswith(b'hipv4-')
    pos += 24+length
end = pos
bundle[142:end-8] = bundle[150:end]
bundle[end-8:end] = bytes(8)
# Validate all descriptors using their declared lengths; keep payloads unchanged.
pos = 32
entries = []
for _ in range(9):
    offset, size, length = struct.unpack_from('<QQQ', bundle, pos)
    assert 0 < length < 256 and offset+size <= len(bundle)
    name = bundle[pos+24:pos+24+length].decode('ascii')
    entries.append(dict(target=name, offset=offset, size=size))
    pos += 24+length
assert pos == end-8
assert bundle[4096:] == raw[start+4096:]
repaired = raw[:start]+bundle
assert len(repaired) == len(raw)
out = root/'rdna2/build/registration-probe'
out.mkdir(parents=True, exist_ok=True)
(out/'repaired.bundle').write_bytes(bundle)
(out/'version-header-repaired.dll').write_bytes(repaired)
manifest = dict(source_sha256=hashlib.sha256(raw).hexdigest(),
    repaired_sha256=hashlib.sha256(repaired).hexdigest(),
    change='compact descriptor header by eight bytes; payloads untouched', entries=entries)
(out/'repair.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
