# Stage 41b BACE1 large-pool input result

## Decision

Stage 41b passed its preregistered technical-readiness gate. Of the 49 frozen
BACE1 receptors, 48 receptors and 48 cognate ligands were prepared in one
common coordinate frame. The minimum required count was 40.

This result authorizes the frozen Stage 41c cognate-redocking gate. It does not
establish pose recovery, enrichment, QUBO superiority, quantum execution, or
quantum advantage.

## Independent checks

- All three declared output hashes match the extracted core archive.
- The two manifests contain 48 unique, order-matched conformer IDs.
- The 336 files referenced by both manifests were checked across the core and
  diagnostic archives: zero files were missing and zero hashes differed.
- All 48 receptor-preparation summaries report `status=ok` and Meeko return
  code 0 without enabling the bad-residue override.
- Only 4I12 and 4I1C use the adjudicated `A:216,420=CYX` template override.
- Raw ligand coordinates match the audited mmCIF coordinates exactly for all
  48 prepared cases. The maximum common-frame RMSD is
  `5.45594555450147e-05` angstrom.
- The common box is centered at `(24.35, 13.96, 23.58)` angstrom with size
  `(30, 22, 22)` angstrom and a minimum observed ligand margin of
  `3.7688` angstrom.

## Preserved failure

`BACE1_6DMI_aligned` (`6DMI_5T5`) remains a technical preparation failure.
Receptor preparation succeeded, but the RCSB ModelServer ligand coordinates
did not match the alternate conformer selected from the mmCIF record. The
structure is counted as one failure among the frozen 49 and cannot be replaced.

## Next gate

Run Uni-Dock 1.1.3 enhanced cognate redocking for the 48 prepared receptors
with seeds `20260801`, `20260802`, and `20260803` (144 GPU batches). A receptor
passes when at least two seeds recover the cognate pose at RMSD at most 2.0
angstrom and its median RMSD is at most 2.0 angstrom. At least 40 of the frozen
49 receptors must pass, with no unresolved warning or pose-integrity failure.
