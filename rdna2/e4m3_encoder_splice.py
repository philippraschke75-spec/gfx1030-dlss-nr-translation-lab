#!/usr/bin/env python3
"""
E4M3 Encoder Splice Tool — Phase 1: Extract & Map
================================================================================

Task: Replace 428 old encoders in k_pre_block + 418 in k_post_block with the
new 11-instruction e4m3 sequence, preserving exact output registers and
performing full live-range analysis per site.

Phase 1: Read both .s files, extract all encoder sites, analyze live ranges,
output JSON map of all splice candidates.

Usage:
  python e4m3_encoder_splice.py --phase extract
  python e4m3_encoder_splice.py --phase splice  (Phase 2: apply changes)
  python e4m3_encoder_splice.py --phase assemble (Phase 3: llvm-mc + ld.lld)
  python e4m3_encoder_splice.py --phase difftest (Phase 4: verify)
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field, asdict

# Config
REPO = Path(__file__).resolve().parent.parent
PREBLOCK_S = REPO / 'build' / 'kernels-hw-scratch' / '_Z21k_pre_block_1h_32_fp89PreParams.s'
POSTBLOCK_S = REPO / 'build' / 'kernels-hw-scratch' / '_Z22k_post_block_1h_32_fp810PostParams.s'
MAP_JSON = REPO / 'e4m3_encoder_map.json'
BACKUP_DIR = REPO / 'build' / 'e4m3_backups'

# New e4m3 encoder sequence (11 VALU + comparisons)
# v_in = input (f16), v_out = output code, v_t = temp
NEW_ENCODER_TEMPLATE = """v_cvt_f32_f16_e64  v_out, |v_in|
v_mul_f32_e32      v_out, 0x03800000, v_out      ; * 2^-120
v_bfe_u32          v_t, v_out, 20, 1
v_add3_u32         v_out, v_out, 0x7ffff, v_t
v_lshrrev_b32_e32  v_out, 20, v_out
v_min_u32_e32      v_out, 0x7e, v_out
v_cmp_u_f16_e32    vcc_lo, v_in, v_in
v_cndmask_b32_e64  v_out, v_out, 0x7f, vcc_lo
v_or_b32_e32       v_t, 0x80, v_out
v_cmp_gt_f16_e32   vcc_lo, 0, v_in
v_cndmask_b32_e32  v_out, v_out, v_t, vcc_lo"""

@dataclass
class RegisterDef:
    line_num: int
    vgpr: int
    is_write: bool

@dataclass
class EncoderSite:
    kernel: str  # 'pre_block' or 'post_block'
    site_index: int  # 0-based occurrence number
    start_line: int  # First line of old encoder (v_cmpx_o_f16 or earlier)
    end_line: int    # Last line of old encoder (s_or_b32 exec_lo restoration)
    nff: int         # Number of Free regs before this site
    old_v_in: int    # Input f16 register
    old_v_out: int   # Output code register (MUST preserve)
    old_v_t: int     # Temp register used
    old_exec_sgpr: int  # SGPR storing exec before encoder
    new_v_in: int = -1  # After live-range analysis
    new_v_out: int = -1
    new_v_t: int = -1
    conflict_reason: str = ""  # Why splice failed (if any)
    live_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # VGPR -> (first_use, last_use)

def load_file(path: Path) -> List[str]:
    """Load .s file, return lines without trailing newline."""
    with open(path, 'r') as f:
        return [line.rstrip('\n') for line in f.readlines()]

def extract_register(operand: str) -> int:
    """Extract VGPR index from 'v123', 'v[123:124]', etc. Return -1 if not a VGPR."""
    m = re.search(r'v(\d+)', operand)
    if m:
        return int(m.group(1))
    return -1

def extract_all_registers(line: str) -> Set[int]:
    """Extract all VGPR indices from a line."""
    regs = set()
    for m in re.finditer(r'v(\d+)', line):
        regs.add(int(m.group(1)))
    return regs

def find_encoder_sites(lines: List[str], kernel_name: str) -> List[EncoderSite]:
    """
    Find all encoder sites in the file.
    Encoder pattern: v_cmpx_o_f16 → ... (encoder body) ... → s_or_b32 exec_lo, exec_lo, s*
    """
    sites = []
    i = 0
    site_index = 0

    while i < len(lines):
        line = lines[i]

        # Look for v_cmpx_o_f16 (start of encoder region)
        if 'v_cmpx_o_f16' in line:
            # Extract the f16 register
            m = re.search(r'v_cmpx_o_f16.*?v(\d+)', line)
            if not m:
                i += 1
                continue
            v_in = int(m.group(1))

            # Scan forward to find the encoder body end
            # End is marked by: s_or_b32 exec_lo, exec_lo, s*
            # and we need to find where the exec is restored
            start_line = i
            j = i + 1
            old_v_out = -1
            old_v_t = -1
            old_exec_sgpr = -1

            # Scan the encoder body: look for the v_ldexp_f32 (start of division) and
            # the final v_or_b32 that writes the output
            while j < len(lines) and j < i + 200:  # encoders are ~100-200 lines
                body_line = lines[j]

                # Pattern: v_min_i32_e32 v<out>, 0x7e, v<out>
                # This is near the end of the encoder
                if 'v_min_i32_e32' in body_line and '0x7e' in body_line:
                    m = re.search(r'v_min_i32_e32\s+v(\d+)', body_line)
                    if m:
                        old_v_out = int(m.group(1))

                # Look for the restoration: s_or_b32 exec_lo, exec_lo, s<n>
                if body_line.strip().startswith('s_or_b32') and 'exec_lo' in body_line and old_v_out > 0:
                    # Found the end
                    m = re.search(r's_or_b32\s+exec_lo,\s+exec_lo,\s+s(\d+)', body_line)
                    if m:
                        old_exec_sgpr = int(m.group(1))
                        end_line = j

                        # We need to find v_t (temp register) — scan back for v_ldexp_f32 or similar
                        # Actually, look for the first register written after v_cmpx_o_f16
                        # (besides the output)
                        for k in range(i+1, min(j, i+50)):
                            k_line = lines[k]
                            if 'v_ldexp_f32' in k_line or 'v_log_f32' in k_line:
                                m = re.search(r'v_(?:ldexp|log)_f32.*?v(\d+)', k_line)
                                if m:
                                    potential_t = int(m.group(1))
                                    if potential_t != v_in and potential_t != old_v_out:
                                        old_v_t = potential_t
                                        break

                        if old_v_t < 0:
                            # Fallback: find any write in the early part
                            for k in range(i+1, min(j, i+30)):
                                regs_written = extract_all_registers(lines[k])
                                for reg in sorted(regs_written):
                                    if reg != v_in and reg != old_v_out:
                                        old_v_t = reg
                                        break
                                if old_v_t >= 0:
                                    break

                        sites.append(EncoderSite(
                            kernel=kernel_name,
                            site_index=site_index,
                            start_line=start_line,
                            end_line=end_line,
                            nff=-1,  # to be filled later
                            old_v_in=v_in,
                            old_v_out=old_v_out,
                            old_v_t=old_v_t if old_v_t >= 0 else -1,
                            old_exec_sgpr=old_exec_sgpr
                        ))
                        site_index += 1
                        i = j + 1
                        break

                j += 1
            else:
                i += 1
        else:
            i += 1

    return sites

def analyze_live_ranges(lines: List[str], sites: List[EncoderSite]) -> None:
    """
    For each encoder site, compute live ranges of all VGPRs in the entire kernel.
    This is a conservative analysis: a register is "live" from its first write to its last read
    in the entire kernel (not within the encoder).
    """
    # Global live range for the entire kernel
    live_ranges_global: Dict[int, Tuple[int, int]] = {}  # VGPR -> (first_line, last_line)

    for line_num, line in enumerate(lines):
        regs = extract_all_registers(line)
        for reg in regs:
            if reg not in live_ranges_global:
                live_ranges_global[reg] = (line_num, line_num)
            else:
                first, _ = live_ranges_global[reg]
                live_ranges_global[reg] = (first, line_num)

    # For each site, determine which registers are free at that location
    for site in sites:
        site.live_ranges = live_ranges_global.copy()

        # A register is "free" at this site if:
        # - It is not used by the encoder itself
        # - It is not live at the site's location (first write or last read is before/after site)
        # (For now, we keep the old registers; a more advanced search would find alternatives)

def extract_and_map(preblock_path: Path, postblock_path: Path) -> Dict:
    """Phase 1: Extract all sites and output map JSON."""
    print("[Phase 1] Loading files...")
    pre_lines = load_file(preblock_path)
    post_lines = load_file(postblock_path)

    print("[Phase 1] Finding encoder sites in k_pre_block...")
    pre_sites = find_encoder_sites(pre_lines, 'pre_block')

    print("[Phase 1] Finding encoder sites in k_post_block...")
    post_sites = find_encoder_sites(post_lines, 'post_block')

    print(f"[Phase 1] Found {len(pre_sites)} sites in k_pre_block, {len(post_sites)} in k_post_block")

    all_sites = pre_sites + post_sites

    print("[Phase 1] Analyzing live ranges...")
    analyze_live_ranges(pre_lines, pre_sites)
    analyze_live_ranges(post_lines, post_sites)

    print("[Phase 1] Writing map JSON...")
    map_data = {
        'metadata': {
            'kernel_files': {
                'pre_block': str(preblock_path),
                'post_block': str(postblock_path),
            },
            'new_encoder_template': NEW_ENCODER_TEMPLATE,
            'target_hash': 'd54273b81de7cf88',
            'reference_frame_ms': 206.6,
        },
        'sites': [asdict(site) for site in all_sites],
        'summary': {
            'total_sites': len(all_sites),
            'pre_block_sites': len(pre_sites),
            'post_block_sites': len(post_sites),
        }
    }

    MAP_JSON.write_text(json.dumps(map_data, indent=2))
    print(f"[Phase 1] Map written to {MAP_JSON}")

    return map_data

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['extract', 'splice', 'assemble', 'difftest'],
                    default='extract', help='Phase to run')
    ap.add_argument('--preblock', type=Path, default=PREBLOCK_S)
    ap.add_argument('--postblock', type=Path, default=POSTBLOCK_S)
    a = ap.parse_args()

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    if a.phase == 'extract':
        extract_and_map(a.preblock, a.postblock)
    else:
        print(f"[Phase {a.phase.upper()}] Not yet implemented")

if __name__ == '__main__':
    main()
