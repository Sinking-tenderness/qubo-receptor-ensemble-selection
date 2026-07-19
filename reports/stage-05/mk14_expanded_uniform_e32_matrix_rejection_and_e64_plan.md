# Stage 5 MAPK14 Uniform e32 Matrix Rejection And e64 Plan

Date: 2026-07-19

## Complete e32 Evidence

The complete uniform e32 result archive
`stage05_mk14_expanded8_train160_e32_core_results_v1.tar.gz` has SHA-256
`0BED344CEEA19FB68CF302ECBC80B12C1A07C9CC4CE71C7DDDB0E1F3FE91A5C5`.
The local hash exactly matched the remote hash.

All three paired-seed runs completed without a failed receptor-ligand pair:

| Seed | Pairs | Failed | Wall time (s) |
|---|---:|---:|---:|
| 20260801 | 1,280 | 0 | 9,379.611 |
| 20260802 | 1,280 | 0 | 9,224.756 |
| 20260803 | 1,280 | 0 | 9,268.246 |

The total sequential wall time was `7.74 h`. No validation or test row entered
the runs or aggregation.

## Consensus Admission

Remote and independent local audits produced identical results:

- aggregate pairs: 1,280;
- nonnegative median-score pairs: 0;
- median median-minus-minimum: `0.018 kcal/mol`;
- maximum median-minus-minimum: `1.347 kcal/mol`;
- pairs passing the frozen two-of-three gate: 1,266;
- pairs failing the gate: 14.

Thus, approximately `98.9%` of the matrix passed, but the preregistered
all-pairs requirement rejected the complete e32 matrix. The 14 failures were
distributed across 1A9U, 2BAJ, 3K3J, 3KQ7, and 3OCG rather than one corrupted
receptor column.

No e16 cell was reused, no e32 cell was selectively replaced, and no
enrichment or QUBO calculation was performed. Machine-readable local outputs:

- `data/stage05_mk14_expanded_e32_matrix_admission_summary.json`;
- `data/processed/stage05_mk14_expanded_e32_matrix_flagged_pairs.csv`.

## e64 Protocol-Selection Diagnostic

The frozen failure action sends all 14 rejected pairs to e64 using the same
three ligand-specific seeds. This creates 42 Vina jobs. Pair selection is based
only on the e32 consensus audit, not ligand labels.

Each e64 case must have:

1. three successful, negative scores;
2. at least two scores within `0.5 kcal/mol` of the pair minimum;
3. median-minus-minimum at most `0.5 kcal/mol`.

If all 14 cases pass, the result supports a future complete uniform e64 matrix
recomputation. It does not rescue or modify the rejected e32 matrix. If any
case fails, QUBO remains closed and the search protocol must be reconsidered.

## Remote Bundle

- file: `stage05_mk14_expanded8_train160_e64_consensus_remote_v1.tar.gz`;
- local path: `D:/量子×蛋白质/stage05_mk14_expanded8_train160_e64_consensus_remote_v1.tar.gz`;
- size: 1,920,238 bytes;
- SHA-256: `AEA50475035D98E9B6A2E15560267AB6B71F2F32038A885EC95D1B3419B5498B`;
- expected jobs: 42;
- expected 32-vCPU wall time: approximately 10 to 20 minutes.

Run on the Linux instance:

```bash
set -euo pipefail

ARCHIVE=/root/autodl-tmp/stage05_mk14_expanded8_train160_e64_consensus_remote_v1.tar.gz
WORK=/root/autodl-tmp/stage05_mk14_expanded8_train160_e64_consensus_v1
LOG=/root/autodl-tmp/stage05_mk14_expanded8_train160_e64_consensus.log

sha256sum "$ARCHIVE"
mkdir -p "$WORK"
tar -xzf "$ARCHIVE" -C "$WORK"
cd "$WORK"
sha256sum -c bundle_manifest.sha256

bash scripts/run_stage05_mk14_expanded_e64_consensus_remote.sh \
  | tee "$LOG"

cp stage05_mk14_expanded8_train160_e64_consensus_diagnostics_v1.tar.gz \
  /root/autodl-tmp/
sha256sum \
  /root/autodl-tmp/stage05_mk14_expanded8_train160_e64_consensus_diagnostics_v1.tar.gz
```

The returned diagnostic archive must be audited before deciding whether a full
e64 matrix is justified.
