# Selected diagnostic evidence

These files preserve the relevant local test evidence without distributing
runtime DLLs or GPU payloads. Paths in logs identify the test machine's files;
module-relative offsets are the useful addresses for comparison.

* `hip-initialization-exceptions.log`: pre-repair first-chance exception stacks.
  A logged exception alone does not prove termination.
* `bundle-repair.json`: original/repaired runtime hashes and corrected table.
* `game-stacks.txt`, `reporter-stacks.txt`: stalled process thread snapshots.
* `STACK_FINDINGS.md`: observations and limits of that inspection.

Historical build-directory links in audit notes refer to the original local
workspace. This folder contains the selected evidence available in GitHub.
