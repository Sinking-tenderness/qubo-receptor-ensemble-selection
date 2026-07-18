# Stage 5 MAPK14 Expanded-Train Matrix Admission And e32 Diagnostic Plan

Date: 2026-07-18

## Completed Docking Evidence

The expanded MAPK14 train-only benchmark completed all three paired e16 seeds:

- receptors: 8;
- ligands: 160 development-train rows, 80 active and 80 decoy;
- receptor-ligand pairs per seed: 1,280;
- total Vina jobs: 3,840;
- failed pairs: 0 in every seed;
- validation rows: 0;
- locked-test rows: 0.

The source result archive is
`stage05_mk14_expanded8_train160_e16_core_results_v1.tar.gz`, SHA-256
`B3BCE3E08421CE1245E4EEF6786C0C97A4512DB1B7A367720354CB387DBAA45D`.

## Independent Matrix Admission

The aggregation, seed summaries, representative score tables, primary median
matrix, and minimum-score sensitivity matrix passed independent hash and shape
checks. The preregistered search-stability gate nevertheless rejected the
matrix before enrichment or QUBO fitting:

| Check | Frozen limit | Observed |
|---|---:|---:|
| Nonnegative seed-score pairs | 0 | 1 |
| Seed range above 2.0 kcal/mol | 0 | 7 |
| Maximum seed range | 2.0 kcal/mol | 9.9703 kcal/mol |

The complete seed-range distribution contained 119 pairs above 0.5 kcal/mol,
50 above 1.0 kcal/mol, and 7 above 2.0 kcal/mol. The median range was 0.058
kcal/mol, so the rejection is driven by a small high-risk tail rather than a
global execution failure.

The largest case was `MK14_active_L000082 / MK14_2BAJ_aligned`: seed0 returned
`+0.6313 kcal/mol`, whereas seed1 and seed2 returned `-9.227` and `-9.339`
kcal/mol. This pattern is consistent with an isolated search failure, but that
interpretation must be tested rather than assumed.

Machine-readable audit outputs:

- `data/stage05_mk14_expanded_train_matrix_admission_summary.json`;
- `data/processed/stage05_mk14_expanded_train_matrix_flagged_pairs.csv`.

## Frozen Diagnostic Decision

Every automatically flagged pair, independent of its ligand label, will be
rerun at e32 with the same three paired seeds. This gives 7 cases and 21 Vina
jobs. The box, scoring function, CPU per job, and single-pose output remain
unchanged; only exhaustiveness increases from 16 to 32.

Every case must satisfy all of the following inherited 0.5 kcal/mol checks:

1. all three e32 runs complete and have negative scores;
2. e32 seed range is at most 0.5 kcal/mol;
3. absolute e16-median versus e32-median difference is at most 0.5 kcal/mol;
4. absolute e16-minimum versus e32-minimum difference is at most 0.5 kcal/mol.

All seven cases must pass. The original e16 median and minimum matrices remain
unchanged; no flagged cell may be selectively replaced by an e32 result. A
failure keeps QUBO fitting, validation sampling, and test release closed.

## Remote Bundle

- file: `stage05_mk14_expanded8_train160_e32_matrix_diagnostics_remote_v1.tar.gz`;
- local path: `D:/量子×蛋白质/stage05_mk14_expanded8_train160_e32_matrix_diagnostics_remote_v1.tar.gz`;
- size: 1,729,713 bytes;
- SHA-256: `ACE0202835B28218C2B9FED891557AD0C578D96E5B8EC40292CBF34B4DF1B220`.

Run on the 32-vCPU Linux instance:

```bash
set -euo pipefail

ARCHIVE=/root/autodl-tmp/stage05_mk14_expanded8_train160_e32_matrix_diagnostics_remote_v1.tar.gz
WORK=/root/autodl-tmp/stage05_mk14_expanded8_train160_e32_matrix_diagnostics_v1

sha256sum "$ARCHIVE"
mkdir -p "$WORK"
tar -xzf "$ARCHIVE" -C "$WORK"
cd "$WORK"
sha256sum -c bundle_manifest.sha256

bash scripts/run_stage05_mk14_expanded_matrix_diagnostics_remote.sh \
  | tee /root/autodl-tmp/stage05_mk14_expanded8_train160_e32_matrix_diagnostics.log

cp stage05_mk14_expanded8_train160_e32_matrix_diagnostics_v1.tar.gz \
  /root/autodl-tmp/
sha256sum /root/autodl-tmp/stage05_mk14_expanded8_train160_e32_matrix_diagnostics_v1.tar.gz
```

With 16 concurrent two-CPU jobs, this targeted diagnostic should normally take
only several minutes. The returned archive must be audited before the train-only
QUBO non-degeneracy gate is opened.

This diagnostic does not establish enrichment, biological activity, affinity
accuracy, QUBO benefit, test performance, or quantum advantage.
