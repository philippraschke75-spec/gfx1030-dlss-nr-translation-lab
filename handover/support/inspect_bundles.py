"""Read-only inspection of embedded Clang offload bundles; never loads the input."""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

MAGIC = b'__CLANG_OFFLOAD_BUNDLE__'

def inspect(path, output):
    data = path.read_bytes()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for match in re.finditer(re.escape(MAGIC), data):
        base = match.start()
        cursor = base + len(MAGIC)
        try:
            count, = struct.unpack_from('<Q', data, cursor)
            cursor += 8
            if not 1 <= count <= 64:
                continue
            entries = []
            for _ in range(count):
                offset, size, name_len = struct.unpack_from('<QQQ', data, cursor)
                cursor += 24
                if not 0 < name_len <= 1024 or cursor + name_len > len(data):
                    raise ValueError('invalid target')
                target = data[cursor:cursor + name_len].decode('ascii')
                cursor += name_len
                if base + offset + size > len(data):
                    raise ValueError('payload outside input')
                entries.append((target, offset, size))
            if any(base + offset < cursor for _, offset, _ in entries):
                continue
            for index, (target, offset, size) in enumerate(entries):
                payload = data[base + offset:base + offset + size]
                name = f'{base:08x}-{index}-' + re.sub(r'[^a-zA-Z0-9_.-]', '_', target)
                is_elf = payload.startswith(b'\x7fELF')
                if is_elf:
                    (output / (name + '.elf')).write_bytes(payload)
                records.append(dict(bundle_offset=base, target=target, size=size,
                                    sha256=hashlib.sha256(payload).hexdigest(),
                                    elf=is_elf, file=name + '.elf' if is_elf else None))
        except (ValueError, UnicodeDecodeError, struct.error):
            continue
    report = dict(input=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), entries=records)
    (output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    inspect(args.input, args.output)
