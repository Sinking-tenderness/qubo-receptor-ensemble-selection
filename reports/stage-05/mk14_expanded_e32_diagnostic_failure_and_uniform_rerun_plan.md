# Stage 5 MAPK14 e32 Diagnostic Failure And Uniform Rerun Plan

Date: 2026-07-18

## Diagnostic Integrity

The targeted e32 result archive
`stage05_mk14_expanded8_train160_e32_matrix_diagnostics_v1.tar.gz` has SHA-256
`8F6F4D8E56B83B8C7EDF05FD47EA467B230025D7E4C85841C63970AFA5214154`.
All 21 expected Vina jobs completed successfully with the frozen box, e32,
two CPUs per job, one output mode, and the paired ligand-specific seeds.

The independent audit reproduced every score and case decision:

- cases rescued: 5 of 7;
- unresolved cases: 2 of 7;
- score cells replaced: 0;
- enrichment metrics calculated: 0;
- QUBO fits: 0;
- validation/test rows read: 0.

The machine-readable audit is
`data/stage05_mk14_expanded_e32_diagnostic_audit_summary.json`.

## Unresolved Search Behavior

Both unresolved cases involve `MK14_active_L000348`:

| Receptor | e16 scores | e32 scores | e32 result |
|---|---|---|---|
| 2BAJ | -10.78, -10.76, -12.94 | -12.95, -12.94, -12.90 | e32 converged, but the e16 median was wrong by 2.16 kcal/mol |
| 3KQ7 | -10.37, -13.14, -11.80 | -11.80, -13.15, -13.18 | two e32 seeds agreed; one remained in an unfavorable alternate basin |

The three 2BAJ e32 poses had a maximum fixed-order heavy-atom RMSD of only
`0.088 A`. For 3KQ7, the unfavorable pose was approximately `10.835 A` from
the two mutually consistent favorable poses. This confirms stochastic basin
selection rather than input corruption.

The unchanged e16 median and minimum matrices remain rejected. Using e32 only
for these two cells, promoting the e16 minimum matrix, or proceeding directly
to QUBO would violate the frozen protocol.

## Uniform e32 Decision

Before any screening metric was calculated, amendment01 froze a complete e32
recomputation of all 8 receptors by 160 train ligands under the same three
paired seeds. The primary matrix remains the three-seed median, and the minimum
matrix remains sensitivity only.

The revised admission rule requires every pair to have at least two of three
scores within `0.5 kcal/mol` of the pair minimum. Equivalently for three seeds,
the median-minus-minimum difference must be at most `0.5 kcal/mol`. One isolated
unfavorable seed is reported but does not reject a pair when the other two
agree. This criterion was frozen before the complete e32 matrix was run.

If any pair fails this consensus gate, QUBO remains closed and all failed pairs
enter label-independent e64 diagnostics. No cell may be selectively replaced.

## Remote Bundle

- file: `stage05_mk14_expanded8_train160_e32_remote_v1.tar.gz`;
- local path: `D:/量子×蛋白质/stage05_mk14_expanded8_train160_e32_remote_v1.tar.gz`;
- size: 2,171,780 bytes;
- SHA-256: `8627B5D73D535288AD5A6345BE9C28AC42E149297BB3180E5A4D1982FEA393E0`;
- expected Vina jobs: 3,840;
- estimated 32-vCPU wall time: approximately 7.6 hours, with 8 to 9 hours reserved.

Run on the Linux instance:

```bash
set -euo pipefail

ARCHIVE=/root/autodl-tmp/stage05_mk14_expanded8_train160_e32_remote_v1.tar.gz
WORK=/root/autodl-tmp/stage05_mk14_expanded8_train160_e32_v1
LOG=/root/autodl-tmp/stage05_mk14_expanded8_train160_e32.log

sha256sum "$ARCHIVE"
mkdir -p "$WORK"
tar -xzf "$ARCHIVE" -C "$WORK"
cd "$WORK"
sha256sum -c bundle_manifest.sha256

nohup bash scripts/run_stage05_mk14_expanded_train_e32_remote.sh \
  > "$LOG" 2>&1 &
echo $! | tee /root/autodl-tmp/stage05_mk14_expanded8_train160_e32.pid
```

Monitor without interrupting the run:

```bash
tail -40 /root/autodl-tmp/stage05_mk14_expanded8_train160_e32.log

ps -p "$(cat /root/autodl-tmp/stage05_mk14_expanded8_train160_e32.pid)" \
  -o pid,lstart,etime,%cpu,%mem,cmd

grep -E "docking receptor|measured_wall_runtime|matrix_admission|SHA256|sha256" \
  /root/autodl-tmp/stage05_mk14_expanded8_train160_e32.log | tail -30
```

After completion:

```bash
WORK=/root/autodl-tmp/stage05_mk14_expanded8_train160_e32_v1

cat "$WORK/results/runs/stage05_mk14_expanded8_train160_e32_matrix_admission/summary.json"

cp "$WORK/stage05_mk14_expanded8_train160_e32_core_results_v1.tar.gz" \
  /root/autodl-tmp/
sha256sum /root/autodl-tmp/stage05_mk14_expanded8_train160_e32_core_results_v1.tar.gz
```

The returned core archive must pass local admission before any train-only QUBO
non-degeneracy calculation is started.
