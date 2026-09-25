"""Experimental static GFX11 -> GFX1030 translation of one identified kernel.

Not a general translator. Generates a separate module; never patches the input.
Assembly success alone is NOT a correctness result.
"""
import argparse
import hashlib
import math
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'research/src/emulator'))
import p14d_kd as kd

SYMBOL = '_Z12k_final_head10HeadParams'
INPUT = ROOT/'analysis/installer-bundles/000e3200-3-hipv4-amdgcn-amd-amdhsa--gfx1100.elf'
EXPECTED = '51a5ac02b52aa2b6f04a0521537833e51b28c96c13d3145a2b45b11d1aa6f1e7'
BIN = Path('C:/Program Files/AMD/ROCm/6.4/bin')
OUT = ROOT/'rdna2/build/final-head'

ALIASES = {
    's_load_b32':'s_load_dword', 's_load_b64':'s_load_dwordx2',
    's_load_b128':'s_load_dwordx4', 's_and_not1_b32':'s_andn2_b32',
    's_or_not1_b32':'s_orn2_b32', 's_and_not1_saveexec_b32':'s_andn2_saveexec_b32',
    'global_load_b64':'global_load_dwordx2', 'global_load_u8':'global_load_ubyte',
    'global_load_u16':'global_load_ushort',
    'global_load_d16_hi_b16':'global_load_short_d16_hi',
    'global_store_b8':'global_store_byte', 'ds_store_b8':'ds_write_b8',
    'ds_store_b32':'ds_write_b32', 'ds_load_b128':'ds_read_b128',
}

def lower_pack_f16(args, scratch):
    """Pack raw halfwords, expanding floating inline constants as FP16 bits.

    Integer ALU instructions otherwise interpret `1.0` as FP32 0x3f800000,
    whose low halfword is zero. Capture both sources before writing the
    destination so destination/source aliases retain parallel-read semantics.
    """
    d, a, b = [x.strip() for x in args.split(',')]
    def bits(src):
        if re.fullmatch(r'-?\d+\.\d+', src):
            return hex(struct.unpack('<H', struct.pack('<e', float(src)))[0])
        return src
    return [f'v_mov_b32_e32 v{scratch}, {bits(a)}',
            f'v_mov_b32_e32 v{scratch+1}, {bits(b)}',
            f'v_and_b32_e32 v{scratch}, 0xffff, v{scratch}',
            f'v_lshl_or_b32 {d}, v{scratch+1}, 16, v{scratch}']


def reg_range(s):
    m = re.fullmatch(r'v\[(\d+):(\d+)\]',s.strip())
    if not m or int(m[2])-int(m[1]) != 7:
        raise ValueError('unexpected WMMA register shape: '+s)
    return int(m[1])

def wmma(text, scratch=84, gather_base=None, dpp_sgpr=None):
    regs = [reg_range(x) for x in text.split(None,1)[1].split(',')]
    d,a,b,c=regs
    if dpp_sgpr is not None:
        # No LDS. The serial form's ds_bpermute address ((lane>>4)<<2) + 8*j reads lane (lane>>4) + 2*j: lanes 0-15
        # take lane 2j, lanes 16-31 take lane 2j+1. Build, per A register k, a vector whose row 0 is A itself and
        # whose row 1 lane i holds A's row-0 lane i+1 (v_permlanex16 with select nibbles min(i+1,15)); then
        # v_dot2c with DPP row_share:2j reads lane 2j of its own row: row 0 -> A[2j], row 1 -> A[2j+1].
        # Same operands, same k order per output register as the serial form, so bit-identical.
        slo, shi, smask = dpp_sgpr, dpp_sgpr+1, dpp_sgpr+2
        tmp = scratch+9
        result=[f's_mov_b32 s{slo}, 0x87654321', f's_mov_b32 s{shi}, 0xffedcba9', f's_mov_b32 s{smask}, 0xffff0000']
        for k in range(8):
            result += [f'v_permlanex16_b32 v{tmp}, v{a+k}, s{slo}, s{shi}',
                       f'v_cndmask_b32_e64 v{gather_base+k}, v{a+k}, v{tmp}, s{smask}']
        for j in range(8):
            result.append(f'v_mov_b32_e32 v{scratch+j}, v{c+j}')
            result += [f'v_dot2c_f32_f16_dpp v{scratch+j}, v{gather_base+k}, v{b+k} row_share:{2*j} row_mask:0xf bank_mask:0xf'
                       for k in range(8)]
        result += [f'v_mov_b32_e32 v{d+j}, v{scratch+j}' for j in range(8)]
        return result
    if gather_base is not None:
        # Batched: for each output register issue its 8 ds_bpermute into 8 distinct gather registers, wait
        # ONCE, then run the 8 v_dot2c in the same k order as the serial form - same accumulation order,
        # so bit-identical, but one LDS round-trip latency per output register instead of eight.
        lane=scratch+8
        result=[f'v_mbcnt_lo_u32_b32 v{lane}, -1, 0',
                f'v_lshrrev_b32_e32 v{lane}, 4, v{lane}',
                f'v_lshlrev_b32_e32 v{lane}, 2, v{lane}']
        for j in range(8):
            result.append(f'v_mov_b32_e32 v{scratch+j}, v{c+j}')
            result += [f'ds_bpermute_b32 v{gather_base+k}, v{lane}, v{a+k} offset:{8*j}' for k in range(8)]
            result.append('s_waitcnt lgkmcnt(0)')
            result += [f'v_dot2c_f32_f16 v{scratch+j}, v{gather_base+k}, v{b+k}' for k in range(8)]
        result += [f'v_mov_b32_e32 v{d+j}, v{scratch+j}' for j in range(8)]
        return result
    # v84..v93 are disjoint from the original kernel's v0..v83.
    # All inputs are read before any original destination is changed.
    lane,gather=scratch+8,scratch+9
    result=[f'v_mbcnt_lo_u32_b32 v{lane}, -1, 0',
            f'v_lshrrev_b32_e32 v{lane}, 4, v{lane}',
            f'v_lshlrev_b32_e32 v{lane}, 2, v{lane}']
    for j in range(8):
        result.append(f'v_mov_b32_e32 v{scratch+j}, v{c+j}')
        for k in range(8):
            result += [f'ds_bpermute_b32 v{gather}, v{lane}, v{a+k} offset:{8*j}',
                       's_waitcnt lgkmcnt(0)',
                       f'v_dot2c_f32_f16 v{scratch+j}, v{gather}, v{b+k}']
    result += [f'v_mov_b32_e32 v{d+j}, v{scratch+j}' for j in range(8)]
    return result

def main():
    global INPUT, BIN, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--abi-probe', action='store_true')
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--hip-bin', type=Path, default=BIN)
    parser.add_argument('--output', type=Path, default=OUT)
    options = parser.parse_args()
    INPUT, BIN, OUT = options.input.resolve(), options.hip_bin.resolve(), options.output.resolve()
    probe = options.abi_probe
    blob, sections, syms = kd.parse_elf(str(INPUT))
    if hashlib.sha256(blob).hexdigest()!=EXPECTED:
        raise ValueError('input hash mismatch')
    OUT.mkdir(parents=True, exist_ok=True)
    ro = next(s for s in sections if s['name']=='.rodata')
    robytes=bytearray(blob[ro['offset']:ro['offset']+ro['size']])
    (OUT/'original-rodata.bin').write_bytes(robytes)
    lut=next(s for s in syms if s['name']=='g_e4m3_lut')
    if lut['size']!=512: raise ValueError('unexpected lookup-table size')
    lutoffset=lut['value']-ro['addr']
    if any(robytes[lutoffset:lutoffset+512]):
        raise ValueError('expected runtime-initialized zero table in source')
    # The extracted device image does not include runtime initialization.
    # Populate the E4M3FN -> IEEE FP16 table required by this isolated module.
    # The original conversion code clamps to 448 and reserves 0x7f for NaN.
    for bits in range(256):
        sign=-1.0 if bits&128 else 1.0
        exponent=(bits>>3)&15; mantissa=bits&7
        if exponent==15 and mantissa==7:
            value=math.nan
        elif exponent==0:
            value=sign*math.ldexp(mantissa,-9)
        else:
            value=sign*math.ldexp(1.0+mantissa/8.0,exponent-7)
        robytes[lutoffset+2*bits:lutoffset+2*bits+2]=struct.pack('<e',value)
    (OUT/'initialized-rodata.bin').write_bytes(robytes)
    dis=subprocess.run([str(BIN/'llvm-objdump.exe'),'--disassemble','--mcpu=gfx1100',str(INPUT)],
                       capture_output=True,text=True,check=True).stdout
    selected=[]; active=False; entry=None
    for line in dis.splitlines():
        label=re.fullmatch(r'([0-9a-fA-F]+) <([^>]+)>:',line)
        if label:
            if active: break
            if label[2]==SYMBOL: active=True;entry=int(label[1],16)
        elif active and '//' in line:
            text, encoded=line.split('//',1)
            addr=re.match(r'\s*([0-9a-fA-F]+):',encoded)
            selected.append((int(addr[1],16),text.strip(),encoded))
    if entry!=0xa3c00: raise ValueError('unexpected kernel entry')
    # This hash-specific lowering reserves v84..v95. Refuse a source whose
    # register footprint would make the replacement overwrite live values.
    used_vgprs=set()
    for _, instruction, _ in selected:
        used_vgprs.update(map(int,re.findall(r'\bv(\d+)\b',instruction)))
        for first,last in re.findall(r'\bv\[(\d+):(\d+)\]',instruction):
            used_vgprs.update(range(int(first),int(last)+1))
    if not used_vgprs or max(used_vgprs)>83:
        raise ValueError('source VGPR footprint overlaps reserved lowering temporaries')
    code=['.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"',
          '.amdhsa_code_object_version 5','.text','.p2align 8','.Loriginal_rodata:',
          '.incbin "'+(OUT/'initialized-rodata.bin').as_posix()+'"',
          '.p2align 8, 0',f'.globl {SYMBOL}',f'.type {SYMBOL},@function',f'{SYMBOL}:',
          's_mov_b32 s15, s2 // map GFX1030 workgroup id into original ABI slot']
    changes=[]
    addresses={a for a,_,_ in selected}
    for i,(address,text,encoded) in enumerate(selected):
        code.append(f'.Lpc_{address:x}:')
        code.append('// original: '+text)
        op, *rest=text.split(None,1)
        args=rest[0] if rest else ''
        result=None
        if op=='v_wmma_f32_16x16x16_f16':
            result=wmma(text)
        elif op=='s_getpc_b64':
            result=[text, f'.Lgetpc_{address:x}:']
        elif i>0 and selected[i-1][1].startswith('s_getpc_b64'):
            m=re.fullmatch(r's_add_u32 (s\d+), \1, (0x[0-9a-f]+)',text)
            if not m: raise ValueError('unrecognized PC-relative sequence')
            signed=int(m[2],16);signed-=0x100000000 if signed>=0x80000000 else 0
            target=address+signed
            if not ro['addr']<=target<ro['addr']+ro['size']:
                raise ValueError('PC-relative target outside constant data')
            result=[f's_add_u32 {m[1]}, {m[1]}, .Loriginal_rodata+{target-ro["addr"]}-.Lgetpc_{selected[i-1][0]:x}']
        elif op.startswith('s_cbranch_') or op=='s_branch':
            target=re.search(r'<'+re.escape(SYMBOL)+r'\+0x([0-9a-f]+)>',encoded)
            if not target: raise ValueError('unresolved branch')
            dest=entry+int(target[1],16)
            if dest not in addresses: raise ValueError('branch into instruction')
            result=[f'{op} .Lpc_{dest:x}']
        elif op in ('s_clause','s_set_inst_prefetch_distance'):
            result=[]  # scheduling hints, not memory/control semantics
        elif op in ('s_delay_alu','s_waitcnt_depctr'):
            result=['s_waitcnt_depctr 0', 's_nop 7']
        elif op=='s_waitcnt':
            result=['s_waitcnt vmcnt(0) lgkmcnt(0)', 's_waitcnt_vscnt null, 0']
        elif op.startswith('v_dual_'):
            pair=text.split(' :: ')
            if len(pair)!=2: raise ValueError('unexpected VOPD')
            results=[]; destinations=[]
            for j,part in enumerate(pair):
                instruction,operands=part.split(None,1)
                if instruction not in ('v_dual_mov_b32','v_dual_add_f32','v_dual_and_b32'):
                    raise ValueError('unhandled VOPD opcode')
                dest,sources=operands.split(',',1)
                destinations.append(dest)
                results.append(instruction.replace('v_dual_','v_')+f' v{94+j},'+sources)
            result=results+[f'v_mov_b32_e32 {d}, v{94+j}' for j,d in enumerate(destinations)]
        elif op=='v_pack_b32_f16':
            result=lower_pack_f16(args, 94)
        elif op in ALIASES:
            if op.startswith('s_load_'): args=args.replace(', null',', 0')
            result=[ALIASES[op]+' '+args]
        else:
            result=[text]
        if result!=[text]: changes.append(dict(address=hex(address),original=text,replacement=result))
        code.extend(result)
    if probe:
        code=code[:code.index(f'{SYMBOL}:')+1]+[
            's_load_dwordx2 s[4:5], s[0:1], 0',
            's_load_dwordx4 s[8:11], s[0:1], 24',
            's_waitcnt lgkmcnt(0)',
            'v_mov_b32_e32 v1, s4', 'v_mov_b32_e32 v2, s5',
            'v_mov_b32_e32 v3, s8', 'v_mov_b32_e32 v4, s9',
            'v_mov_b32_e32 v5, s10', 'v_mov_b32_e32 v6, s11',
            'v_cmpx_eq_u32_e32 0, v0',
            'global_store_dwordx4 v[1:2], v[3:6], off',
            's_endpgm']
    code += [f'.size {SYMBOL}, .-{SYMBOL}', '.section .rodata', '.p2align 6',
             f'.amdhsa_kernel {SYMBOL}',
             '.amdhsa_group_segment_fixed_size 8192', '.amdhsa_private_segment_fixed_size 0',
             '.amdhsa_kernarg_size 280', '.amdhsa_user_sgpr_count 2',
             '.amdhsa_user_sgpr_private_segment_buffer 0',
             '.amdhsa_user_sgpr_kernarg_segment_ptr 1',
             '.amdhsa_system_sgpr_workgroup_id_x 1',
             '.amdhsa_system_sgpr_workgroup_id_y 0', '.amdhsa_system_sgpr_workgroup_id_z 0',
             '.amdhsa_system_sgpr_private_segment_wavefront_offset 0',
             '.amdhsa_system_vgpr_workitem_id 0', '.amdhsa_next_free_vgpr 96',
             '.amdhsa_next_free_sgpr 18', '.amdhsa_reserve_vcc 1',
             '.amdhsa_wavefront_size32 1', '.amdhsa_float_round_mode_32 0',
             '.amdhsa_float_round_mode_16_64 0', '.amdhsa_float_denorm_mode_32 3',
             '.amdhsa_float_denorm_mode_16_64 3', '.amdhsa_dx10_clamp 1', '.amdhsa_ieee_mode 1',
             '.end_amdhsa_kernel']
    # Always obtain metadata from the same hash-checked ELF as the code.
    # A saved dump could silently belong to a different source artifact.
    notes=subprocess.run([str(BIN/'llvm-readobj.exe'),'--notes',str(INPUT)],
                         capture_output=True,text=True,check=True,timeout=20).stdout
    (OUT/'source-notes.txt').write_text(notes,encoding='utf-8')
    pos=notes.index('.name:           '+SYMBOL)
    start=notes.rfind('  - .args:',0,pos)
    end=notes.find('  - .args:',pos)
    if start<0 or end<0:
        raise ValueError('expected delimited kernel metadata record not found')
    md=notes[start:end]
    md=re.sub(r'\.vgpr_count:\s+\d+', '.vgpr_count:     96',md)
    code += ['.amdgpu_metadata','---','amdhsa.version: [1, 2]','amdhsa.kernels:',md,'...','.end_amdgpu_metadata']
    stem='abi_probe' if probe else 'final_head'
    source=OUT/(stem+'.s')
    source.write_text('\n'.join(code)+'\n')
    (OUT/(stem+'-translation.json')).write_text(json.dumps(dict(source_sha256=EXPECTED,changes=changes,
       status='EXPERIMENTAL: requires static and physical validation'),indent=2))
    proc=subprocess.run([str(BIN/'llvm-mc.exe'),'-triple=amdgcn-amd-amdhsa',
                         '-mcpu=gfx1030','-filetype=obj',str(source),'-o',str(OUT/(stem+'.o'))],
                        capture_output=True,text=True,timeout=20)
    (OUT/(stem+'-assembly.log')).write_text(proc.stdout+proc.stderr)
    print(proc.stderr[:12000])
    if proc.returncode: return proc.returncode
    subprocess.run([str(BIN/'ld.lld.exe'),'-shared',str(OUT/(stem+'.o')),'-o',str(OUT/(stem+'.co'))],check=True,timeout=20)
    (OUT/(stem+'-build.json')).write_text(json.dumps(dict(
        source_sha256=EXPECTED,
        translator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        module_sha256=hashlib.sha256((OUT/(stem+'.co')).read_bytes()).hexdigest(),
        source_max_vgpr=max(used_vgprs),temporary_vgprs=[84,95],
        abi_probe=probe,metadata_source='llvm-readobj --notes on hash-checked input',
        status='BUILD_ONLY: no physical result implied'),indent=2)+'\n',encoding='utf-8')
    print('Assembled candidate (not yet validated):',OUT/(stem+'.co'))
    return 0

if __name__=='__main__':
    sys.exit(main())
