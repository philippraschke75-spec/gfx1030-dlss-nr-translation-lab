"""Static address audit for SWINvar32true's source-PC 0xc3908 store only.

Derived from 0xc383c..0xc3968; assumes the path is enabled. This is not
the full VarParams ABI or proof that another store path is race-free.
"""
import argparse
import json
from collections import defaultdict


def trunc_half(n):
    return n // 2 if n >= 0 else -((-n) // 2)


def audit(height, width, grid=(1, 1), offset_x=-4, offset_y=-4):
    h, w = trunc_half(height), trunc_half(width)
    writes = defaultdict(list)
    negative_coordinates = 0
    for gy in range(grid[1]):
        for gx in range(grid[0]):
            for tid in range(512):
                y = trunc_half(offset_y + 8 * gy) + (tid >> 7)
                x = trunc_half(offset_x + 8 * gx) + ((tid >> 5) & 3)
                if y >= h or x >= w:
                    continue
                negative_coordinates += int(y < 0 or x < 0)
                addr = 16 * (y * w + x) + (tid & 15) + (tid & 16) * h * w
                writes[addr].append(dict(group=[gx, gy], wave=(tid % 256) // 32,
                                         lane=tid % 32, iteration=tid // 256,
                                         x=x, y=y, channel=tid & 31))
    collisions = [{'offset': a, 'writers': v} for a, v in writes.items() if len(v) > 1]
    return dict(scope='source store 0xc3908, assuming enabled', height=height,
                width=width, grid=grid, offset_x=offset_x, offset_y=offset_y,
                half_shape=[h, w], plane_bytes=h*w*16, negative_coordinates=negative_coordinates,
                before_base_bytes=sum(a < 0 for a in writes),
                collision_bytes=len(collisions), collisions=collisions)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('height', type=int)
    p.add_argument('width', type=int)
    p.add_argument('--offset-x', type=int, default=-4)
    p.add_argument('--offset-y', type=int, default=-4)
    p.add_argument('--gx', type=int, default=1)
    p.add_argument('--gy', type=int, default=1)
    a = p.parse_args()
    print(json.dumps(audit(a.height, a.width, (a.gx, a.gy), a.offset_x, a.offset_y), indent=2))
