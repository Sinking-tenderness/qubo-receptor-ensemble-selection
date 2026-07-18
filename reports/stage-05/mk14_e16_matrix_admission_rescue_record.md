# Stage 5 MAPK14 e16 Matrix Admission Rescue Record

Date: 2026-07-18

## Scope

The three-seed e16 MAPK14 development benchmark completed all 2,880 docking
jobs and aggregated 960 receptor-ligand pairs. This record resolves the matrix
admission failure caused by two raw seed ranges above the preregistered
`2.0 kcal/mol` screening threshold.

No ROC-AUC, PR-AUC, BEDROC, EF, receptor selection, or QUBO calculation was
performed before this decision. The diagnostic pair selection and all rescue
checks were label-blind. The locked test partition remained absent.

## Source Integrity

- e16 aggregate archive SHA-256:
  `98210382896DFA84CE26F8755A7F86C0F7C938672757B1262051637CEE7798AF`
- e32 diagnostic archive SHA-256:
  `04F54BA529CC0A07D9CA4263065EFF871DA6CE5ACAEB935E8BC3D123D3C49F98`
- Aggregated score rows: 960
- Unique receptor-ligand pairs: 960
- Failed pairs across all three e16 seeds: 0
- Per-seed search warnings: 0
- Nonnegative scores: 0
- Independent aggregation recomputation errors: 0
- Locked-test rows: 0

All three seed summaries, representative score tables, and aggregate outputs
matched the SHA-256 values recorded in the aggregate summary.

## Initial Raw-Range Gate

The raw e16 seed-range distribution was:

| Statistic | Value (kcal/mol) |
|---|---:|
| Median | 0.058 |
| 90th percentile | 0.483 |
| 95th percentile | 0.830 |
| 99th percentile | 1.652 |
| Maximum | 2.297 |

Ninety-four pairs exceeded `0.5 kcal/mol`, 33 exceeded `1.0 kcal/mol`, and two
exceeded `2.0 kcal/mol`. The preregistered raw-range gate therefore failed and
triggered a label-blind e32 search diagnostic before any enrichment metric was
read.

## e32 Diagnostic Results

The rescue acceptance threshold was `0.5 kcal/mol`, inherited from
`configs/stage05_mk14_search_ladder_multiseed_followup.json`, whose threshold
was frozen before the 240-ligand development matrix was run.

Every flagged pair had to satisfy all three checks:

1. e32 three-seed range at most `0.5 kcal/mol`;
2. absolute e16-median versus e32-median difference at most `0.5 kcal/mol`;
3. absolute e16-minimum versus e32-minimum difference at most `0.5 kcal/mol`.

| Pair | e16 range | e32 range | Median delta | Minimum delta | Result |
|---|---:|---:|---:|---:|---|
| L000290 / 2QD9 | 2.297 | 0.198 | 0.033 | 0.001 | pass |
| L000491 / 3KQ7 | 2.160 | 0.025 | 0.009 | 0.007 | pass |

For L000290, e16 seeds 0 and 1 agreed within `0.066 kcal/mol`, while seed 2
was an isolated unfavorable search result. For L000491, e16 seeds 1 and 2
agreed within `0.020 kcal/mol`, while seed 0 was the isolated unfavorable
result. The three e32 runs converged in both cases.

As a supplementary geometric check, the maximum fixed-frame, symmetry-aware
pairwise RMSD among e32 poses was `1.186 A` for L000290 and `0.422 A` for
L000491. Pose RMSD was not used as a rescue acceptance criterion because the
development matrix contains scores rather than pose labels.

## Decision

Status: `matrix_admission_rescued`.

The unchanged e16 median matrix and unchanged e16 minimum sensitivity matrix
are authorized for the preregistered train/validation analysis. No score cell
was deleted, replaced, averaged with e32, or manually corrected.

A full e32 rerun is not required at this gate because:

- both raw-range failures converged at e32;
- e16 robust median and minimum values already match the e32 results within
  `0.033` and `0.007 kcal/mol`, respectively;
- the original three-seed aggregation was designed to resist one failed seed;
- replacing only flagged cells would violate the frozen matrix contract; and
- rerunning the full matrix would add computation without changing the
  validated robust aggregate values.

This decision authorizes development analysis only. It does not establish
enrichment, receptor complementarity, QUBO benefit, biological activity,
locked-test performance, or quantum advantage.

## Reproducibility

- Audit configuration:
  `configs/stage05_mk14_e16_matrix_rescue_audit.json`
- Audit implementation:
  `scripts/audit_stage05_e32_matrix_rescue.py`
- Machine-readable result:
  `data/stage05_mk14_e16_matrix_admission_rescue_summary.json`
- Result SHA-256:
  `51283A625594618078E81AEC9E239D64DF684FD05F1BE24BF14FF673FB492731`
