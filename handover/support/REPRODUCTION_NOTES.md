# Supporting local experiment

These are copies of original local research scripts and retained logs. They are not merged upstream. Generated vendor data is intentionally absent. Read the limits in `rdna2/FINAL_HEAD_RESULTS.md` before reusing the result.

The scripts expect this working layout under the current directory:

```text
inspect_bundles.py
rdna2/                         (included original experiment scripts)
research/src/emulator/          (obtain from the pinned upstream checkout)
analysis/installer-bundles/     (generate from your own matching installer)
analysis/gfx1100-notes.txt       (generate locally)
```

Acquire the upstream source as `research` and check out commit `dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8`. The experiment imports `p14d_kd` from its `src/emulator` directory. Original scripts assume ROCm 6.4 at the standard Windows installation path; review and explicitly configure another path if needed. Python 3.10+ and MSVC/Windows SDK are prerequisites. Do not confuse Python's availability with the Windows `py` launcher's availability.

Host-only input preparation from this directory:

```powershell
python inspect_bundles.py 'PATH-TO-YOUR/dlssnr_on_amd_setup.exe' analysis/installer-bundles
```

The supplied version's installer hash must be:

```text
cf7ada1486b499700a84846b342ca2b1defdb4db622843f812151f255f2ad63c
```

The required extracted ELF is `analysis/installer-bundles/000e3200-3-hipv4-amdgcn-amd-amdhsa--gfx1100.elf`; its hash must match the main handover. Generate metadata with ROCm 6.4 `llvm-readobj.exe --notes` against that file and save its textual output to `analysis/gfx1100-notes.txt`. Use ASCII/UTF-8 compatible text, not PowerShell 5's default UTF-16 redirection. The translator itself invokes `llvm-objdump.exe --disassemble --mcpu=gfx1100`; a preexisting full disassembly is not required for the translation command.

For inspection helpers only, `inspect_kernel.py` and `kernel_inventory.py` additionally expect the disassembly files named in those scripts. They are not prerequisites of `translate_final_head.py`.

Then the generator commands in the main handover create a standalone ABI probe or final-head module. Generation runs assembler/linker tools but does not launch a GPU kernel. The PowerShell runner commands are physical GPU tests; run them separately under the repository's hardware-testing procedure. Each runner invocation builds a test and performs its selected bounded dispatch, with no automatic retry.

The old failed logs are deliberate evidence, not the expected current result:

- `final-head-abi.log`: invalid module-function attribute query before dispatch; corrected in the subsequent source/log.
- `final-head-constant.log`: missing runtime LUT initialization in the isolated module; corrected in `final-head-initialized.log` and subsequent runs.

The original local README mentions Cyberpunk because that was the earlier testing objective. Neither Cyberpunk nor GTA was tested; the current handover's target is GTA V Enhanced.

Do not publish generated ELF, assembly, code objects, embedded headers, weights or copied runtime binaries as part of this source contribution. Review the upstream third-party notices and preserve its license/provenance records for reused upstream source.
