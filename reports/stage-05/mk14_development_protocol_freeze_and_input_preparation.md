# Stage 5 MAPK14 Development Protocol Freeze and Input Preparation

Date: 2026-07-17

## Scope

This checkpoint freezes the MAPK14 development docking protocol before
development-scale receptor comparison or validation metrics are observed. It
also records preparation of a balanced train/validation development panel and
the pre-execution audit of three paired Vina seed replicates.

The locked test partition was not sampled, prepared, docked, or evaluated. Its
status remains `locked_unreleased`.

## Train-Only Search Calibration

Four high-risk train-only receptor-ligand cases were run at exhaustiveness 8,
16, and 32 with seed `20260726`. All 12 runs succeeded. The largest absolute
e16-to-e32 top-score difference was `0.162 kcal/mol`; the other three cases had
no difference at the reported precision.

Two cases received a targeted paired-seed follow-up at e16 and e32 with seeds
`20260727` and `20260728`:

| Case | e32 seed range (kcal/mol) | Maximum e16-e32 difference (kcal/mol) |
|---|---:|---:|
| flexible active L000318 on 2QD9 | 0.040 | 0.095 |
| charged decoy L000178 on 1A9U | 0.074 | 0.220 |

All eight follow-up runs succeeded. Both cases remained below the prespecified
`0.5 kcal/mol` instability threshold, so e16 was selected as the lowest tested
search strength that passed this calibration.

The single-seed and follow-up summary SHA-256 values are respectively
`04E08BFF6355EDBCAAD4F91EC04F3BACE3DA942A6FA1211106BD9641308B896E`
and
`9588C9269380CC5C7CAF9BDDD375EC0CE808F0865A926AFBCA077F1CD9EDD6B2`.

This is a bounded search-strength calibration, not proof that e16 finds a
global optimum for every ligand. It also does not establish enrichment,
receptor complementarity, biological activity, or a QUBO benefit.

## Frozen Development Protocol

- Vina version: 1.2.7
- scoring function: Vina
- common box center: `(-0.49, 3.26, 21.83)` A
- common box size: `(22, 24, 30)` A
- exhaustiveness: 16
- output modes per pair: 1
- CPU threads per Vina process: 2
- paired base seeds: `20260801`, `20260802`, `20260803`
- primary seed aggregation: median representative score
- sensitivity aggregation: minimum representative score
- representative score: pose rank 1

The frozen Vina configuration has SHA-256
`97CCBE12C38825D849C91BED4C8C7111564A24C38A7397CA2EDE8B19509A3F6E`.
Search box, exhaustiveness, seeds, score definition, development ligand
identities, and aggregation rules must not be changed after development
metrics are observed.

## Development Panel

The panel was selected from the frozen grouped scaffold/source-ID split without
using test rows.

| Role | Active | Decoy | Total | Unique split groups/scaffolds |
|---|---:|---:|---:|---:|
| train | 80 | 80 | 160 | 160 |
| validation | 40 | 40 | 80 | 80 |
| combined | 120 | 120 | 240 | 240 |

Train is reserved for receptor-selection and QUBO-objective fitting.
Validation is reserved for hyperparameter and development-method comparison.
It must not be used to fit the objective. Test remains reserved for a separate,
one-time release decision.

## Ligand Preparation Audit

- RDKit 3D embedding succeeded for 240/240 ligands.
- MMFF94 converged within 500 iterations for 229 ligands.
- Eleven ligands embedded successfully but retained a non-convergence warning.
- Meeko PDBQT preparation succeeded for 240/240 ligands.
- All 240 PDBQT rows contain a file SHA-256 value.
- Forty-seven ligands have nonzero formal charge.
- No prepared file, class, or role-count failures were detected.

The 11 warnings are retained rather than silently excluded. They identify
ligands that may need sensitivity inspection if they later produce anomalous
docking behavior; they are not preparation failures.

## Pre-Execution Gate

All three remote Linux seed configurations passed `--audit-only`:

| Seed run | Base seed | Expected pairs | Test rows | Vina jobs started |
|---|---:|---:|---:|---:|
| seed0 | 20260801 | 960 | 0 | 0 |
| seed1 | 20260802 | 960 | 0 | 0 |
| seed2 | 20260803 | 960 | 0 | 0 |

Each run contains four receptors and 240 ligands, for 960 receptor-ligand pairs.
The planned remote layout uses 16 concurrent Vina processes, two CPU threads
per process, at least 32 CPU threads and preferably at least 64 GB RAM. The
official Vina executable is CPU-based, so a GPU is not required.

## Decision and Next Gate

The development input and protocol gate passes. The next step is to execute the
three seed runs with checkpoint resume enabled, aggregate each receptor-ligand
score by the frozen median rule, and calculate train and validation metrics
separately. Any receptor-subset or QUBO objective must be fitted on train only.

The post-run aggregation entry point is
`scripts/aggregate_seed_replicates.py` with configuration
`configs/stage05_mk14_development_seed_aggregation.json`. It rejects incomplete
or failed seed matrices, changed run configurations, mismatched output hashes,
unexpected ligand roles, and any test row before writing the primary median
matrix and minimum-score sensitivity matrix.

No MAPK14 development enrichment, receptor superiority, subset benefit, QUBO
advantage, or test performance is claimed at this checkpoint.
