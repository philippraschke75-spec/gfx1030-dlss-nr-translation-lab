"""Extract the original on-disk bundle for registration-only diagnostics."""
from pathlib import Path
import hashlib
import json
import struct

root = Path(__file__).resolve().parent.parent
source = root / 'gfx1032-dlss-nr-research-main/version.dll'
backup = source.with_name('version.before-bundle-header-repair.dll')
if backup.exists():
    source = backup
data = source.read_bytes()
digest = hashlib.sha256(data).hexdigest()
if digest != 'fa616204dd68521217875fb41835d3189c89e7c87c3050e069df16bafc458a05':
    raise ValueError('Runtime changed; review before extracting')
magic = b'__CLANG_OFFLOAD_BUNDLE__'
start = data.index(magic)
pos = start + len(magic)
count, = struct.unpack_from('<Q', data, pos)
pos += 8
if not 0 < count < 64:
    raise ValueError('Unexpected bundle count')
entries = []
end = 0
parse_error = None
for _ in range(count):
    offset, size, length = struct.unpack_from('<QQQ', data, pos)
    pos += 24
    if length > 4096 or start + offset + size > len(data):
        parse_error = dict(descriptor_offset=pos-24-start, offset=offset,
                           size=size, name_length=length)
        break
    name = data[pos:pos+length].decode('ascii')
    pos += length
    entries.append(dict(target=name, offset=offset, size=size))
    end = max(end, offset+size)
if not parse_error and start+end < pos:
    raise ValueError('Invalid header extent')
out = root / 'rdna2/build/registration-probe'
out.mkdir(parents=True, exist_ok=True)
# Preserve malformed bytes exactly for reproduction; no repair in this probe.
bundle = data[start:] if parse_error else data[start:start+end]
(out/'original.bundle').write_bytes(bundle)
manifest = dict(source_sha256=digest, file_offset=start,
               bundle_sha256=hashlib.sha256(bundle).hexdigest(), entries=entries,
               parse_error=parse_error)
(out/'source.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
