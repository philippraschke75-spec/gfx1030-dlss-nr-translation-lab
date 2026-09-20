"""Original synthetic conformance kernel; builds only, uses no vendor input."""
import json
from pathlib import Path
import translate_kernels as t

out=t.base.ROOT/'rdna2/build/lowering-probe';out.mkdir(parents=True,exist_ok=True)
name='lowering_probe'
instructions=[
    's_mov_b32 s4, s2', 's_load_b64 s[0:1], s[0:1], 0',
    'v_lshl_add_u32 v1, s4, 8, v0',
    'v_xor_b32 v2, 0x11223344, v1', 'v_add_nc_u32 v3, 0x01020304, v1',
    'v_xor_b32 v4, -1, v1', 'v_mov_b32 v5, 0x89abcdef',
    'scratch_store_b128 off, v[1:4], off', 'scratch_store_b32 off, v5, off offset:16',
    's_waitcnt vmcnt(0)', 'v_and_b32 v6, 3, v0', 'v_lshlrev_b32 v6, 2, v6',
    'scratch_load_b32 v7, v6, off', 'v_and_b32 v6, 3, v0',
    'scratch_load_u8 v8, v6, off offset:16', 'scratch_load_b32 v9, off, off offset:8',
    's_waitcnt vmcnt(0)', 'v_lshlrev_b32 v10, 5, v1',
    'v_add_co_u32 v20, vcc_lo, s0, v10',
    'v_add_co_ci_u32_e64 v21, null, s1, 0, vcc_lo',
    'global_store_b32 v[20:21], v7, off offset:4096',
    's_cmp_eq_u32 s4, 0',
    'global_store_b32 v10, v8, s[0:1] offset:4100',
    's_cselect_b32 s5, 31, 63', 'v_mov_b32 v11, s5',
    'v_add_co_u32 v22, vcc_lo, 8192, v20',
    'v_add_co_ci_u32_e64 v23, null, 0, v21, vcc_lo',
    'global_store_b32 v[22:23], v9, off offset:-4088',
    'global_store_b32 v[20:21], v11, off offset:4108',
    'v_and_b32 v12, 31, v0', 'v_lshlrev_b32 v13, v12, 1',
    'v_cmp_eq_u32 vcc_lo, 0, v12', 'v_cndmask_b32 v13, v13, 0, vcc_lo',
    'v_clz_i32_u32_e32 v14, v13',
    'global_store_b32 v[20:21], v14, off offset:4112',
    's_endpgm',
]
lines=[(i*4,text,'') for i,text in enumerate(instructions)]
policy={'private':24,'dyn_stack':0,'user_enables':[('kernarg_segment_ptr',2)],
        'system_enables':[('workgroup_id_x',1),('private_segment_wavefront_offset',1)],
        'user_count':2,'group':0,'kernarg_size':8,'workitem_vgpr':0,'dx10':1,'ieee':1}
md='''  - .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name: lowering_probe
    .private_segment_fixed_size: 24
    .sgpr_count: 6
    .symbol: lowering_probe.kd
    .vgpr_count: 24
    .wavefront_size: 32
'''
rofile=out/'dummy-rodata.bin';rofile.write_bytes(bytes(16))
result=t.translate(name,lines,policy,md,{'addr':0,'size':16},rofile,out,t.base.BIN,private_lds=True)
(out/'build.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
