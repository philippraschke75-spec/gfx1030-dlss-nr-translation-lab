"""Decode a blocked-launch record without dereferencing captured pointers."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def inspect(path):
    record = json.loads(Path(path).read_text())
    if record.get('schema') != 1 or record.get('kind') != 'blocked_first_swin_launch':
        raise ValueError('unsupported capture schema')
    if record.get('args_read_ok') is not True:
        raise ValueError('capture did not read argument bytes successfully')
    raw = bytes.fromhex(record['args_hex'])
    if len(raw) != 168 or record.get('explicit_bytes') != 168:
        raise ValueError('expected exactly 168 explicit argument bytes')
    h, w, x, y = struct.unpack_from('<iiii', raw, 0x18)
    return dict(symbol=record['symbol'], grid=record['grid'], block=record['block'],
                args_sha256=hashlib.sha256(raw).hexdigest(),
                inferred_fields=dict(height=h, width=w, offset_x=x, offset_y=y,
                                     flags=struct.unpack_from('<I', raw, 0x28)[0]),
                words=[dict(offset=hex(i), value=hex(struct.unpack_from('<I', raw, i)[0]))
                       for i in range(0, 168, 4)],
                replay_ready=False,
                missing=['payload identity validation', 'allocation map and pointer relocations',
                         'device buffers and weights', 'complete dispatch sequence and dependencies'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    print(json.dumps(inspect(p.parse_args().capture), indent=2))
