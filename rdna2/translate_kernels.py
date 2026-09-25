"""Build isolated gfx1030 candidates and an honest coverage ledger. No GPU work.

Only the pinned source image is accepted. Scratch/call kernels are refused
until their ABI is implemented. Assembly success is not numerical validation.
"""
import argparse, os as _os
TX_DELAY=_os.environ.get('TX_DELAY','none')
TX_WAITCNT=_os.environ.get('TX_WAITCNT','faithful')
TX_WMMA=_os.environ.get('TX_WMMA','dpp')
TX_DIVPOW2=_os.environ.get('TX_DIVPOW2','1')=='1'
import hashlib
import json
import math
import re
import struct
import subprocess
from pathlib import Path
import translate_final_head as base
import p14d_dec as dec

ALIASES = dict(base.ALIASES, **{
    's_load_b256':'s_load_dwordx8', 's_load_b512':'s_load_dwordx16',
    'global_load_b32':'global_load_dword', 'global_load_b96':'global_load_dwordx3',
    'global_load_b128':'global_load_dwordx4', 'global_store_b16':'global_store_short',
    'global_store_b32':'global_store_dword', 'global_store_b64':'global_store_dwordx2',
    'global_store_b96':'global_store_dwordx3', 'global_store_b128':'global_store_dwordx4',
    'global_load_b16':'global_load_ushort', 'ds_load_b32':'ds_read_b32',
    'global_load_i8':'global_load_sbyte', 'global_load_i16':'global_load_sshort',
    'ds_load_b64':'ds_read_b64', 'ds_store_b64':'ds_write_b64',
    'ds_store_b128':'ds_write_b128', 'ds_load_u8':'ds_read_u8',
    'ds_load_u16':'ds_read_u16', 'ds_store_b16':'ds_write_b16',
    'ds_load_2addr_b32':'ds_read2_b32', 'ds_store_2addr_b32':'ds_write2_b32',
    'ds_load_2addr_b64':'ds_read2_b64', 'ds_store_2addr_b64':'ds_write2_b64',
    'ds_load_2addr_stride64_b32':'ds_read2st64_b32',
    'ds_store_2addr_stride64_b32':'ds_write2st64_b32',
    'ds_load_u16_d16_hi':'ds_read_u16_d16_hi',
    'ds_store_b16_d16_hi':'ds_write_b16_d16_hi',
    'global_atomic_cmpswap_b32':'global_atomic_cmpswap',
    'global_load_u16':'global_load_ushort','flat_load_b32':'flat_load_dword',
    'flat_load_u16':'flat_load_ushort','ds_store_b32':'ds_write_b32',
    's_and_not1_b32':'s_andn2_b32','s_or_not1_b32':'s_orn2_b32',
    's_and_not1_saveexec_b32':'s_andn2_saveexec_b32','ds_store_b96':'ds_write_b96','ds_load_b96':'ds_read_b96',
    'v_dot2acc_f32_f16':'v_dot2c_f32_f16',
})
HELPER='_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti'

def run(args):
    return subprocess.run([str(x) for x in args],capture_output=True,text=True,timeout=25,check=True).stdout

def functions(dis):
    result={}; current=None
    for line in dis.splitlines():
        label=re.fullmatch(r'([0-9a-fA-F]+) <([^>]+)>:',line)
        if label:
            current=[];result[label[2]]=current
        elif current is not None and '//' in line:
            text,raw=line.split('//',1)
            addr=re.match(r'\s*([0-9a-fA-F]+):',raw)
            if addr: current.append((int(addr[1],16),text.strip(),raw))
    return result

def register_end(lines,kind):
    found=[]
    for _,text,_ in lines:
        found += [int(x) for x in re.findall(r'\b'+kind+r'(\d+)\b',text)]
        found += [int(b) for a,b in re.findall(r'\b'+kind+r'\[(\d+):(\d+)\]',text)]
    return max(found,default=-1)+1

def metadata(notes,name):
    for item in notes.split('  - .args:')[1:]:
        if re.search(r'^    \.name:\s+'+re.escape(name)+r'\s*$',item,re.M):
            # Last record is followed by amdhsa.version and YAML terminator.
            item=re.split(r'^amdhsa\.|^\.\.\.',item,flags=re.M)[0]
            return '  - .args:'+item
    raise ValueError('metadata not found')

def value(md,key):
    m=re.search(r'^    \.'+key+r':\s+(\d+)',md,re.M)
    if not m: raise ValueError('missing metadata '+key)
    return int(m[1])

def reads_initial_sgpr(lines,reg):
    """Conservative CFG check: is the incoming SGPR read before overwritten?"""
    def contains(operand):
        if re.search(r'\bs'+str(reg)+r'\b',operand): return True
        return any(int(a)<=reg<=int(b) for a,b in re.findall(r'\bs\[(\d+):(\d+)\]',operand))
    byaddr={a:i for i,(a,_,_) in enumerate(lines)};pending=[0];seen=set()
    while pending:
        i=pending.pop()
        if i in seen or i>=len(lines): continue
        seen.add(i);addr,text,_=lines[i];op,*tail=text.split(None,1);args=tail[0] if tail else ''
        first,_,rest=args.partition(',')
        writes=((op.startswith('s_') and not op.startswith(('s_cmp','s_cbranch','s_branch','s_wait','s_nop','s_barrier','s_end','s_send','s_store','s_set','s_clause','s_delay')))
                or op.startswith(('v_readfirstlane','v_readlane'))
                or (op.startswith('v_cmp_') and '_e64' in op))
        if contains(rest if writes else args): return True
        if writes and contains(first): continue
        if op=='s_endpgm': continue
        if op=='s_branch' or op.startswith('s_cbranch_'):
            imm=int(args,0) if args.startswith('0x') else int(args);imm-=65536 if imm>=32768 else 0
            if addr+4+imm*4 not in byaddr: return True
            pending.append(byaddr[addr+4+imm*4])
            if op=='s_branch': continue
        pending.append(i+1)
    return False

def lower_private(text,base_reg,address_reg):
    """Lower the observed flat private-array forms into disjoint per-thread LDS.

    The caller reserves a byte region and stable base for each local thread.
    Scalar addressing and unsupported modifiers are refused.
    """
    match=re.fullmatch(r'(scratch_(?:load|store)_(?:b32|b64|b128|u8)) (.+)',text)
    if not match: raise ValueError('unsupported private instruction '+text)
    op,args=match.groups()
    m=re.fullmatch(r'([^,]+), ([^,]+), off(?: offset:(\d+))?',args)
    if not m: raise ValueError('unsupported private operand form '+text)
    first,second,offset=m[1].strip(),m[2].strip(),int(m[3] or 0)
    store='store' in op;voffset=first if store else second;data=second if store else first
    if voffset!='off' and not re.fullmatch(r'v\d+',voffset): raise ValueError('unsupported private address')
    if offset>65535: raise ValueError('private immediate out of range')
    prefix=[];addr=f'v{base_reg}'
    if voffset!='off': prefix=[f'v_add_nc_u32 v{address_reg}, v{base_reg}, {voffset}'];addr=f'v{address_reg}'
    suffix=op.split('_',2)[2]
    instruction=f'ds_{"write" if store else "read"}_{suffix}'
    operands=f'{addr}, {data}' if store else f'{data}, {addr}'
    return prefix+[f'{instruction} {operands} offset:{offset}']


DIV_SHAPE = ['v_div_scale_f32','v_div_scale_f32','v_rcp_f32_e32','v_fma_f32','v_fmac_f32_e32','v_mul_f32_e32',
             'v_fma_f32','v_fmac_f32_e32','v_fma_f32','v_div_fmas_f32']
_HINT = ('s_delay_alu','s_waitcnt_depctr')
_READS_DEST = ('v_fmac_','v_mac_','v_dot2c_','v_fmaak','v_fmamk')
_IMPLICIT_VCC_READ = ('v_cndmask_b32_e32','v_add_co_ci_u32_e32','v_sub_co_ci_u32_e32','v_subrev_co_ci_u32_e32','v_div_fmas_f32')

def _ops(text):
    op,*t=text.split(None,1); return op,[x.strip() for x in (t[0].split(',') if t else [])]


def _liveness(lines,targets,regs,helper_entry=None):
    """Backward liveness over the CFG for a set of register names, in the compiler's own model: a write kills, a read
    makes live. s_swappc/s_setpc (call/return) and anything unrecognised make every tracked register live."""
    n=len(lines); addr_idx={a:k for k,(a,_,_) in enumerate(lines)}
    call_sites=[k for k,(a,t,_) in enumerate(lines) if t.split()[0]=='s_swappc_b64']
    if helper_entry is not None and any(t.split()[0]=='s_swappc_b64' for a,t,_ in lines[helper_entry:]): helper_entry=None
    succ=[]; use=[]; kill=[]
    for k,(addr,text,_) in enumerate(lines):
        o,a=_ops(text)
        if o=='s_branch' or o.startswith('s_cbranch_'):
            imm=int(a[0],0) if a[0].startswith('0x') else int(a[0]); imm=imm-65536 if imm>=32768 else imm
            t=addr_idx.get(addr+4+4*imm)
            sc=[t] if t is not None else []
            if o.startswith('s_cbranch_') and k+1<n: sc.append(k+1)
            succ.append(sc); use.append(set()); kill.append(set()); continue
        if o in ('s_endpgm',): succ.append([]); use.append(set()); kill.append(set()); continue
        if o in ('s_swappc_b64','s_setpc_b64'):
            # Interprocedural, for the single-helper layout (helper appended after the caller): a call continues at
            # the helper's entry, a return at the instruction after every call site. Anything else: all live.
            if helper_entry is not None and o=='s_swappc_b64':
                succ.append([helper_entry]); use.append(set()); kill.append(set()); continue
            if helper_entry is not None and o=='s_setpc_b64' and k>=helper_entry and call_sites:
                succ.append([c+1 for c in call_sites if c+1<n]); use.append(set()); kill.append(set()); continue
            succ.append([k+1] if (o=='s_swappc_b64' and k+1<n) else []); use.append(set(regs)); kill.append(set()); continue
        succ.append([k+1] if k+1<n else [])
        if o.startswith(_HINT): use.append(set()); kill.append(set()); continue
        text_regs=set(re.findall(r'\bv\d+\b|\bvcc_lo\b|\bvcc\b',text))
        rng=set()
        for m in re.finditer(r'v\[(\d+):(\d+)\]',text):
            rng|={f'v{r}' for r in range(int(m[1]),int(m[2])+1)}
        dest=a[0] if a else ''
        dest_set={dest}|({f'v{r}' for r in range(int(re.match(r'v\[(\d+):(\d+)\]',dest)[1]),int(re.match(r'v\[(\d+):(\d+)\]',dest)[2])+1)} if re.match(r'v\[(\d+):(\d+)\]',dest) else set())
        if dest=='vcc': dest_set|={'vcc_lo'}
        u=(text_regs|rng)&regs
        kl=set()
        if o.startswith('v_') and not o.startswith(('v_cmpx',)) and a and not o.startswith(_READS_DEST) and not o.startswith(('v_readlane','v_writelane')):
            srcs=set()
            for x in a[1:]:
                srcs|=set(re.findall(r'\bv\d+\b|\bvcc_lo\b|\bvcc\b',x))
                for m in re.finditer(r'v\[(\d+):(\d+)\]',x): srcs|={f'v{r}' for r in range(int(m[1]),int(m[2])+1)}
            if o in _IMPLICIT_VCC_READ: srcs.add('vcc_lo')
            if o.startswith('v_div_scale') and len(a)>1 and a[1]=='vcc_lo': kl.add('vcc_lo'); srcs.discard('vcc_lo')
            u=srcs&regs
            kl|=(dest_set&regs)-srcs
            if o.startswith('v_cmp_') and o.endswith('_e32'): kl.add('vcc_lo')
        elif o.startswith('v_cmpx') or o.startswith('s_') or o.startswith(('ds_','global_','buffer_','scratch_','flat_')):
            if o.startswith(('ds_','global_','buffer_','scratch_','flat_')) and 'load' in o and a:
                dv=set(re.findall(r'\bv\d+\b',a[0]))|{f'v{r}' for m in re.finditer(r'v\[(\d+):(\d+)\]',a[0]) for r in range(int(m[1]),int(m[2])+1)}
                srcs=(text_regs|rng)-dv
                u=srcs&regs; kl=dv&regs
            elif o.startswith('v_cmpx') and o.endswith('_e32'): u=(text_regs|rng)&regs
        else:
            u=set(regs)   # unknown form: be conservative
        use.append(u); kill.append(kl)
    live=[set() for _ in range(n)]
    changed=True
    while changed:
        changed=False
        for k in range(n-1,-1,-1):
            out=set()
            for t in succ[k]: out|=live[t]
            new=use[k]|(out-kill[k])
            if new!=live[k]: live[k]=new; changed=True
    return live

def find_divpow2(lines,helper_entry=None):
    """Power-of-two divisions in the e4m3 encoder that can become one v_ldexp, bit-exactly.
    Returns {fixup_index: (vD, vX, vE)} and the set of line indices whose code is dropped. A site qualifies only if:
    the divisor comes from 'v_ldexp_f32 vP, 1.0, vE'; the division has exactly the IEEE sequence shape DIV_SHAPE;
    no branch target lies inside it; the dividend was clamped to <= 448 (v_cndmask vX, 0x43e00000, ...) and exec was
    restricted to vX >= 2^-6 (v_cmpx_ngt_f32 0x3c800000, vX), with vX, vE, vP unchanged since; and every temporary the
    division writes (VGPRs and vcc) is provably overwritten before any read, without crossing a label or branch.
    Then 2^e and vX/2^e are normal floats, so the division is exact and equal to ldexp(vX, -e) in any denorm mode."""
    targets=set()
    for addr,text,_ in lines:
        op,args=_ops(text)
        if op.startswith('s_cbranch_') or op=='s_branch':
            imm=int(args[0],0) if args[0].startswith('0x') else int(args[0]); imm=imm-65536 if imm>=32768 else imm
            targets.add(addr+4+4*imm)
    targets.add(lines[0][0])
    real=[i for i,(a,t,_) in enumerate(lines) if not t.split()[0].startswith(_HINT)]
    pos={i:k for k,i in enumerate(real)}
    sites={}; drop=set(); cand=[]
    for k,i in enumerate(real):
        op,args=_ops(lines[i][1])
        if op!='v_div_fixup_f32' or len(args)!=4: continue
        vD,vQ,vP,vX=args
        if k<len(DIV_SHAPE): continue
        blk=real[k-len(DIV_SHAPE):k]
        if [ _ops(lines[j][1])[0] for j in blk ]!=DIV_SHAPE: continue
        a0=_ops(lines[blk[0]][1])[1]; a1=_ops(lines[blk[1]][1])[1]
        if a0[2:]!=[vP,vP,vX] or a1[2:]!=[vX,vP,vX] or a0[1]!='null' or a1[1]!='vcc_lo': continue
        if _ops(lines[blk[-1]][1])[1][0]!=vQ: continue
        if any(lines[j][0] in targets for j in range(blk[0],i+1)): continue
        # divisor producer, guard, and no intervening writes to vX/vE/vP
        ld=None; guard=clamp=False; bad=False; vE=None
        for kk in range(k-len(DIV_SHAPE)-1, max(-1,k-len(DIV_SHAPE)-40), -1):
            j=real[kk]; o,a=_ops(lines[j][1])
            if ld is None:
                if a and a[0]==vP:
                    if o=='v_ldexp_f32' and a[1]=='1.0': ld=j; vE=a[2]
                    else: bad=True; break
                continue
            if lines[j][0] in targets and not guard: pass
            if o=='v_cmpx_ngt_f32_e32' and a==['0x3c800000',vX]: guard=True; continue
            if guard and o=='v_cndmask_b32_e32' and a[0]==vX and a[1]=='0x43e00000': clamp=True; break
            if a and a[0] in (vX,) and not o.startswith('v_cmp'): bad=True; break
        if bad or ld is None or not (guard and clamp): continue
        for j in range(ld+1,i):
            o,a=_ops(lines[j][1])
            if a and a[0] in (vE,vP) and not o.startswith(('v_cmp','s_')): bad=True; break
        if bad: continue
        temps={_ops(lines[j][1])[1][0] for j in blk}|{'vcc_lo'}
        temps.discard(vD)
        cand.append((i,vD,vX,vE,blk,temps))
    if not cand: return sites,drop
    live_in=_liveness(lines,targets,set().union(*(c[5] for c in cand)),helper_entry)
    for i,vD,vX,vE,blk,temps in cand:
        if i+1<len(lines) and (live_in[i+1] & temps): continue
        sites[i]=(vD,vX,vE); drop.update(j for j in blk)
    return sites,drop

def translate(name,lines,policy,md,ro,rofile,out,binpath,private_lds=False,helper=None,hw_scratch=False):
    ops={text.split()[0] for _,text,_ in lines}
    if policy['dyn_stack']: raise ValueError('UNSUPPORTED: dynamic stack ABI')
    private=policy['private']
    hw=bool(private and hw_scratch and not private_lds)
    if private and not (private_lds or hw):
        raise ValueError('UNSUPPORTED: private scratch; explicit --private-lds experiment required')
    if ops & {'s_setpc_b64','s_swappc_b64'}:
        if helper is None: raise ValueError('UNSUPPORTED: indirect call without helper body')
        helper_addr=helper[0][0];targets=set()
        for i,(addr,text,raw) in enumerate(lines):
            if not text.startswith('s_swappc_b64'): continue
            for j in range(i-1,max(-1,i-24),-1):
                if not lines[j][1].startswith('s_getpc_b64') or j+1>=i: continue
                m=re.search(r's_add_u32 s\d+, s\d+, (0x[0-9a-f]+)',lines[j+1][1])
                if not m: continue
                delta=int(m[1],16);delta-=0x100000000 if delta>=0x80000000 else 0
                targets.add(lines[j][0]+4+delta);break
            else: raise ValueError('UNSUPPORTED: unresolved indirect call at '+hex(addr))
        if targets!={helper_addr}: raise ValueError('UNSUPPORTED: indirect targets '+','.join(map(hex,targets)))
        if any(t.startswith('s_setpc_b64') for _,t,_ in lines): raise ValueError('UNSUPPORTED: kernel body has its own s_setpc')
        lines=lines+helper
        ops={text.split()[0] for _,text,_ in lines}
    if value(md,'wavefront_size')!=32: raise ValueError('UNSUPPORTED: non-wave32')
    users=dict(policy['user_enables']); systems=dict(policy['system_enables'])
    if set(users)-{'kernarg_segment_ptr','dispatch_ptr'}:
        raise ValueError('UNSUPPORTED user SGPR inputs')
    if set(systems)-{'workgroup_id_x','workgroup_id_y','workgroup_id_z','private_segment_wavefront_offset'}:
        raise ValueError('UNSUPPORTED system SGPR inputs')
    original_system_count=len(systems)
    if private:
        # This mapping handles private array accesses, not direct reads of the
        # runtime's original scratch-wave offset SGPR.
        slot=policy['user_count']+sum(k.startswith('workgroup_id_') for k in systems)
        if reads_initial_sgpr(lines,slot):
            raise ValueError('UNSUPPORTED: incoming scratch-offset SGPR is read before definition')
        systems.pop('private_segment_wavefront_offset',None)
    private_stride=(private+15)//16*16 if not hw else 0
    private_start=(policy['group']+15)//16*16
    lds=private_start+private_stride*value(md,'max_flat_workgroup_size') if private and not hw else policy['group']
    if lds>65536: raise ValueError('UNSUPPORTED: private-to-LDS allocation exceeds 64 KiB')
    scratch=(max(register_end(lines,'v'),value(md,'vgpr_count'))+3)//4*4
    wide_offsets=any(t.startswith('global_') and (m:=re.search(r' offset:(-?\d+)',t)) and not -2048<=int(m[1])<=2047 for _,t,_ in lines)
    vgprs=scratch+(16 if wide_offsets else (14 if private else 12))
    # TX_WMMA=batched puts the 8 WMMA gather registers at scratch+16..+23, clear of every other scratch use.
    wmma_batched=TX_WMMA in ('batched','dpp') and any(t.startswith('v_wmma') for _,t,_ in lines)
    if wmma_batched: vgprs=max(vgprs,scratch+24)
    if vgprs>256: raise ValueError('UNSUPPORTED: register budget exceeds 256 VGPRs')
    user_count=sum(users.values())+(2 if hw else 0)
    # Metadata count may include special registers; next_free_sgpr describes
    # explicitly addressed ordinary SGPRs (s0..s105 on RDNA).
    sgprs=max(register_end(lines,'s'),policy['user_count']+original_system_count)
    carry_sgpr=sgprs
    address_sgpr=(sgprs+1)//2*2
    saved_scc=address_sgpr+2
    if wide_offsets: sgprs=saved_scc+1
    dpp_sgpr=None
    if TX_WMMA=='dpp' and any(t.startswith('v_wmma') for _,t,_ in lines) and saved_scc+4<=106:
        # permlanex16 selects + half-wave mask. A kernel without 3 spare SGPRs (k_conv_res2) keeps the batched
        # LDS lowering, which is bit-identical too.
        dpp_sgpr=saved_scc+1; sgprs=max(sgprs,dpp_sgpr+3)
    if sgprs>106: raise ValueError('UNSUPPORTED: SGPR allocation exceeds supported range')
    prefix=['.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"',
            '.amdhsa_code_object_version 5','.text','.p2align 8','.Loriginal_rodata:',
            '.incbin "'+rofile.as_posix()+'"','.p2align 8, 0',
            f'.globl {name}',f'.type {name},@function',f'{name}:']
    # User enables retain ABI order. Compact system IDs are copied into the
    # original body's slots before any body instruction can clobber them.
    for i in reversed(range(len(systems))):
        prefix.append(f's_mov_b32 s{policy["user_count"]+i}, s{user_count+i}')
    if hw:
        fl=user_count-2;woff=user_count+len(systems)
        prefix += [f's_add_u32 s{fl}, s{fl}, s{woff}',f's_addc_u32 s{fl+1}, s{fl+1}, 0',
                   f's_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s{fl}',f's_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s{fl+1}']
    if private and not hw:
        prefix += [f'v_mul_lo_u32 v{scratch+12}, v0, {private_stride}',
                   f'v_add_nc_u32 v{scratch+12}, {private_start}, v{scratch+12}']
    code=prefix; changes=[]; addresses={a for a,_,_ in lines}; helper_add_at=set()
    div_sites,div_drop=find_divpow2(lines,(len(lines)-len(helper)) if helper is not None and lines[-len(helper):]==helper else None) if TX_DIVPOW2 else ({},set())
    for i,(addr,text,raw) in enumerate(lines):
        code += [f'.Lpc_{addr:x}:','// original: '+text]
        op,*tail=text.split(None,1); args=tail[0] if tail else ''
        replacement=[text]
        if i in div_drop: replacement=[]
        elif i in div_sites:
            vD,vX,vE=div_sites[i]
            replacement=[f'v_sub_nc_u32_e32 v{scratch+10}, 0, {vE}',f'v_ldexp_f32 {vD}, {vX}, v{scratch+10}']
        elif op.startswith('v_wmma'):
            if op!='v_wmma_f32_16x16x16_f16': raise ValueError('unsupported WMMA form '+op)
            replacement=base.wmma(text,scratch,scratch+16 if wmma_batched else None,dpp_sgpr)
        elif op=='s_getpc_b64': replacement=[text,f'.Lgetpc_{addr:x}:']
        elif op=='s_addc_u32' and i and lines[i-1][0] in helper_add_at:
            m=re.fullmatch(r's_addc_u32 (s\d+), (s\d+), (-1|0xffffffff)',text)
            if m and m[1]!=m[2]: m=None
            if not m: raise ValueError('unexpected high-word add after helper PC computation: '+text)
            replacement=[f's_addc_u32 {m[1]}, {m[1]}, 0']
        elif i and lines[i-1][1].startswith('s_getpc_b64'):
            m=re.fullmatch(r's_add_u32 (s\d+), \1, (0x[0-9a-f]+)',text)
            if not m: raise ValueError('unsupported PC-relative sequence')
            offset=int(m[2],16);offset-=0x100000000 if offset>=0x80000000 else 0
            target=addr+offset
            if helper is not None and target==helper[0][0]:
                # The helper is appended after the caller, so the delta is positive here
                # (it was negative in the source); the following s_addc adds only the carry.
                replacement=[f's_add_u32 {m[1]}, {m[1]}, .Lpc_{target:x}-.Lgetpc_{lines[i-1][0]:x}']
                helper_add_at.add(addr)
            else:
                if not ro['addr']<=target<ro['addr']+ro['size']:
                    raise ValueError('unsupported PC-relative reference outside rodata')
                if offset>=0: raise ValueError('unsupported positive PC delta carry policy')
                replacement=[f's_add_u32 {m[1]}, {m[1]}, .Loriginal_rodata+{target-ro["addr"]}-.Lgetpc_{lines[i-1][0]:x}']
        elif op.startswith('s_cbranch_') or op=='s_branch':
            # Branch immediate is a signed count of dwords after this instruction.
            imm=int(args,0) if args.startswith('0x') else int(args)
            imm=imm-65536 if imm>=32768 else imm
            target=addr+4+4*imm
            if target not in addresses: raise ValueError('branch target outside selected function')
            replacement=[f'{op} .Lpc_{target:x}']
        elif op in ('s_clause','s_set_inst_prefetch_distance'): replacement=[]
        # TX_DELAY / TX_WAITCNT select the lowering of RDNA3 scheduling hints and memory waits. Defaults are
        # 'none' (drop s_delay_alu/s_waitcnt_depctr: RDNA2 interlocks ordinary VALU dependencies in hardware)
        # and 'faithful' (keep the original counters). 'faithful' is sound only while every translated
        # instruction issues exactly as many VMEM and LGKM ops as its original - checked for all frame kernels
        # (0 deviations; WMMA expansions end in lgkmcnt(0), the prologue has no memory ops). 'safe'/'full' is
        # the old lowering (full drain + s_nop 7 per hint, full wait per s_waitcnt), kept for A/B runs.
        # Full frame: 561 -> 285-324 ms GPU with TX_WMMA=batched, bit-identical 1.4 GB arena (FRAME_STATE Update 23).
        # TX_WMMA=dpp (default) replaces the LDS gathers with v_permlanex16 + DPP row_share (FRAME_STATE Update 24).
        elif op in ('s_delay_alu','s_waitcnt_depctr'):
            replacement={'safe':['s_waitcnt_depctr 0','s_nop 7'],'none':[],'nop0':['s_nop 0']}[TX_DELAY]
        elif op=='s_waitcnt':
            replacement=([f's_waitcnt {args}'] if TX_WAITCNT=='faithful' and args else
                         ['s_waitcnt vmcnt(0) lgkmcnt(0)','s_waitcnt_vscnt null, 0'])
        elif op.startswith('v_dual_'):
            parts=text.split(' :: '); replacement=[]; dests=[]
            if len(parts)!=2: raise ValueError('invalid dual operation')
            for j,part in enumerate(parts):
                dual,operands=part.split(None,1); dest,src=operands.split(',',1); dests.append(dest)
                normal=dual.replace('v_dual_','v_')
                if normal not in {'v_mov_b32','v_add_f32','v_mul_f32','v_and_b32','v_add_nc_u32','v_cndmask_b32','v_sub_f32','v_subrev_f32','v_max_f32','v_min_f32','v_lshlrev_b32','v_fmac_f32','v_fmamk_f32','v_fmaak_f32'}:
                    raise ValueError('unsupported VOPD '+dual)
                if normal=='v_cndmask_b32': src+=', vcc_lo'
                if normal=='v_fmac_f32': normal='v_fma_f32';src+=', '+dest
                replacement.append(f'{normal} v{scratch+10+j},'+src)
            replacement += [f'v_mov_b32_e32 {d}, v{scratch+10+j}' for j,d in enumerate(dests)]
        elif op=='v_pack_b32_f16':
            replacement=base.lower_pack_f16(args, scratch+10)
        elif op=='v_minmax_i32':
            # LLVM IntMinMaxPat: max(min(src0, src1), src2).
            d,x,y,z=[x.strip() for x in args.split(',')]
            if not all(re.fullmatch(r'(?:[vs]\d+|-?\d+|0x[0-9a-f]+)',v) for v in [d,x,y,z]):
                raise ValueError('unsupported minmax modifiers')
            replacement=[f'v_min_i32 v{scratch+10}, {x}, {y}',f'v_max_i32 {d}, v{scratch+10}, {z}']
        elif op=='v_clz_i32_u32_e32':
            d,x=[x.strip() for x in args.split(',')]
            replacement=[f'v_ffbh_u32 v{scratch+10}, {x}',f'v_min_u32 {d}, 32, v{scratch+10}']
        elif op.startswith('scratch_'):
            if not private: raise ValueError('scratch instruction without private allocation')
            if hw:
                m=re.fullmatch(r'scratch_(load|store)_(b32|b64|b96|b128|u8|i8|u16|i16|b16) (.+)',text)
                if not m: raise ValueError('unsupported private instruction '+text)
                width=dict(b32='dword',b64='dwordx2',b96='dwordx3',b128='dwordx4',u8='ubyte',i8='sbyte',u16='ushort',i16='sshort',b16='short')[m[2]]
                replacement=[f'scratch_{m[1]}_{width} {m[3]}']
            else: replacement=lower_private(text,scratch+12,scratch+13)
        elif op=='s_sendmsg_rtn_b64':
            m=re.fullmatch(r'(s\[\d+:\d+\]), sendmsg\(MSG_RTN_GET_REALTIME\)',args)
            if not m: raise ValueError('UNSUPPORTED: sendmsg_rtn message '+args)
            # Constant-frequency 64-bit counter; result is SMEM, source waits on lgkmcnt.
            replacement=[f's_memrealtime {m[1]}']
        elif op=='s_sendmsg' and args=='sendmsg(MSG_DEALLOC_VGPRS)':
            if i+1>=len(lines) or lines[i+1][1]!='s_endpgm':
                raise ValueError('VGPR deallocation not immediately before program end')
            replacement=[]  # GFX11 early release hint; normal end releases regs.
        elif op in ALIASES:
            if op.startswith('s_load_'): args=args.replace(', null',', 0')
            replacement=[ALIASES[op]+' '+args]
        if op.startswith('global_') and (m:=re.search(r' offset:(-?\d+)',text)) and not -2048<=int(m[1])<=2047:
            # gfx11 has wider offsets. Materialize a full address without
            # clobbering the program's VCC, SCC or original address pair.
            imm=int(m[1]); instruction=replacement[0]
            native,operands=instruction.split(None,1);operands=[x.strip() for x in operands.split(',')]
            address_index=0 if op.startswith('global_store_') else 1
            address=re.fullmatch(r'v\[(\d+):(\d+)\]',operands[address_index])
            scalar=re.match(r's\[(\d+):(\d+)\]',operands[2])
            if len(replacement)!=1: raise ValueError('compound wide-offset operation')
            if scalar:
                if int(scalar[2])!=int(scalar[1])+1: raise ValueError('invalid scalar address pair')
                operands[2]=re.sub(r's\[\d+:\d+\]',f's[{address_sgpr}:{address_sgpr+1}]',operands[2])
                instruction=re.sub(r' offset:-?\d+','',native+' '+', '.join(operands))
                replacement=[f's_cselect_b32 s{saved_scc}, 1, 0',
                             f's_add_u32 s{address_sgpr}, s{scalar[1]}, {imm & 0xffffffff}',
                             f's_addc_u32 s{address_sgpr+1}, s{scalar[2]}, {-1 if imm<0 else 0}',
                             f's_cmp_lg_u32 s{saved_scc}, 0',instruction]
            else:
                if not address or int(address[2])!=int(address[1])+1 or not operands[2].startswith('off'):
                    raise ValueError('unsupported wide-offset address form: '+instruction)
                lo=int(address[1]);temp=scratch+14
                operands[address_index]=f'v[{temp}:{temp+1}]'
                instruction=re.sub(r' offset:-?\d+','',native+' '+', '.join(operands))
                replacement=[f'v_add_co_u32 v{temp}, s{carry_sgpr}, {imm & 0xffffffff}, v{lo}',
                             f'v_add_co_ci_u32_e64 v{temp+1}, null, v{lo+1}, {-1 if imm<0 else 0}, s{carry_sgpr}',instruction]
        if replacement!=[text]: changes.append({'address':hex(addr),'original':text,'replacement':replacement})
        code+=replacement
    code += [f'.size {name}, .-{name}', '.section .rodata','.p2align 6',f'.amdhsa_kernel {name}',
             f'.amdhsa_group_segment_fixed_size {lds}', f'.amdhsa_private_segment_fixed_size {private if hw else 0}',
             f'.amdhsa_kernarg_size {policy["kernarg_size"]}',f'.amdhsa_user_sgpr_count {user_count}',
             '.amdhsa_user_sgpr_private_segment_buffer 0',
             f'.amdhsa_user_sgpr_dispatch_ptr {int("dispatch_ptr" in users)}',
             '.amdhsa_user_sgpr_kernarg_segment_ptr 1']
    if hw: code.append('.amdhsa_user_sgpr_flat_scratch_init 1')
    for axis in 'xyz': code.append(f'.amdhsa_system_sgpr_workgroup_id_{axis} {int("workgroup_id_"+axis in systems)}')
    code += [f'.amdhsa_system_sgpr_private_segment_wavefront_offset {int(hw)}',
             f'.amdhsa_system_vgpr_workitem_id {policy["workitem_vgpr"]}',
             f'.amdhsa_next_free_vgpr {vgprs}',f'.amdhsa_next_free_sgpr {sgprs}',
             '.amdhsa_reserve_vcc 1','.amdhsa_wavefront_size32 1',
             '.amdhsa_float_round_mode_32 0','.amdhsa_float_round_mode_16_64 0',
             '.amdhsa_float_denorm_mode_32 3','.amdhsa_float_denorm_mode_16_64 3',
             f'.amdhsa_dx10_clamp {policy["dx10"]}',f'.amdhsa_ieee_mode {policy["ieee"]}',
             '.end_amdhsa_kernel']
    md=re.sub(r'\.vgpr_count:\s+\d+',f'.vgpr_count:     {vgprs}',md)
    md=re.sub(r'\.sgpr_count:\s+\d+',f'.sgpr_count:     {sgprs}',md)
    md=re.sub(r'\.group_segment_fixed_size:\s+\d+',f'.group_segment_fixed_size: {lds}',md)
    md=re.sub(r'\.private_segment_fixed_size:\s+\d+',f'.private_segment_fixed_size: {private if hw else 0}',md)
    code+=['.amdgpu_metadata','---','amdhsa.version: [1, 2]','amdhsa.kernels:',md,'...','.end_amdgpu_metadata']
    asm=out/(name+'.s');obj=out/(name+'.o');module=out/(name+'.co')
    asm.write_text('\n'.join(code)+'\n',encoding='utf-8')
    (out/(name+'-changes.json')).write_text(json.dumps(changes,indent=2),encoding='utf-8')
    run([binpath/'llvm-mc.exe','-triple=amdgcn-amd-amdhsa','-mcpu=gfx1030','-filetype=obj',asm,'-o',obj])
    run([binpath/'ld.lld.exe','-shared',obj,'-o',module])
    decoded=run([binpath/'llvm-objdump.exe','--disassemble','--mcpu=gfx1030',module])
    # Inspect only the selected function: copied original rodata is not code.
    body=functions(decoded).get(name,[])
    if not body or any('v_wmma' in t or '<unknown>' in t for _,t,_ in body):
        raise ValueError('post-link instruction decode failed')
    return dict(status='ASSEMBLED_UNVALIDATED',module=str(module),sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                source_vgprs=scratch,vgprs=vgprs,lds=lds,private_lds_bytes=private_stride*value(md,'max_flat_workgroup_size'),hw_scratch_bytes=private if hw else 0,
                launch_contract='one-dimensional blocks; local y=z=1' if private and not hw else 'original launch contract',
                wmma_sites=sum(t.startswith('v_wmma') for _,t,_ in lines),
                changed_instructions=len(changes),physical_test='NOT_RUN')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--symbol',action='append',help='exact symbol; omit to inventory/build all kernels')
    ap.add_argument('--input',type=Path,default=base.INPUT)
    ap.add_argument('--output',type=Path,default=base.ROOT/'rdna2/build/kernels')
    ap.add_argument('--hip-bin',type=Path,default=base.BIN)
    ap.add_argument('--hw-scratch',action='store_true',help='native gfx10 flat scratch for private arrays')
    ap.add_argument('--private-lds',action='store_true',help='experimental private arrays in per-thread LDS; build only')
    a=ap.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    blob,sections,syms=base.kd.parse_elf(str(a.input))
    if hashlib.sha256(blob).hexdigest()!=base.EXPECTED: raise ValueError('source hash mismatch')
    dis=functions(run([a.hip_bin/'llvm-objdump.exe','--disassemble','--mcpu=gfx1100',a.input]))
    notes=run([a.hip_bin/'llvm-readobj.exe','--notes',a.input])
    ro=next(s for s in sections if s['name']=='.rodata')
    data=bytearray(blob[ro['offset']:ro['offset']+ro['size']])
    lut=next(s for s in syms if s['name']=='g_e4m3_lut');off=lut['value']-ro['addr']
    if lut['size']!=512 or any(data[off:off+512]): raise ValueError('unexpected LUT identity')
    for bits in range(256):
        e=(bits>>3)&15;m=bits&7;sign=-1 if bits&128 else 1
        v=math.nan if e==15 and m==7 else sign*(math.ldexp(m,-9) if e==0 else math.ldexp(1+m/8,e-7))
        data[off+2*bits:off+2*bits+2]=struct.pack('<e',v)
    rofile=a.output/'initialized-rodata.bin';rofile.write_bytes(data)
    rows=[]
    for name,offset in base.kd.find_kernel_kds(blob,sections,syms):
        if a.symbol and name not in a.symbol: continue
        row={'symbol':name}
        try:
            row.update(translate(name,dis[name],dec.dec(base.kd.dump_kd(blob,offset)),metadata(notes,name),ro,rofile,a.output,a.hip_bin,a.private_lds,dis.get(HELPER),a.hw_scratch))
        except (ValueError,subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            detail=getattr(error,'stderr',None) or str(error)
            (a.output/(name+'-error.txt')).write_text(detail,encoding='utf-8')
            row.update(status='BLOCKED',reason=detail[:4000],physical_test='NOT_RUN')
        rows.append(row);print(name,row['status'],flush=True)
    report={'source_sha256':base.EXPECTED,'translator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'wmma_lowering_sha256':hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
            'warning':'Isolated candidates, not a registered full module; rodata initialization is standalone only.',
            'kernels':rows}
    (a.output/'coverage.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return 0 if rows and all(r['status']=='ASSEMBLED_UNVALIDATED' for r in rows) else 2

if __name__=='__main__': raise SystemExit(main())
