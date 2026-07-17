# Stage 5 MAPK14 Development Method Gate Preregistration

Date: 2026-07-17

## Purpose

This record freezes the MAPK14 development evaluation before the 2,880 planned
Vina jobs are run and before any enrichment metric is observed. Its purpose is
to prevent search failures, validation reuse, or an attractive isolated metric
from being converted into a post hoc QUBO success claim.

The machine-readable contract is
`configs/stage05_mk14_development_method_gate_preregistration.json`.

## Data Boundary

- Train: 80 active and 80 decoy ligands.
- Validation: 40 active and 40 decoy ligands.
- Test: `locked_unreleased` and prohibited.
- Receptors: 2QD9, 1A9U, 3K3J, and 3KQ7 prepared conformers.
- Seed replicates: three frozen e16 runs, each containing 960 pairs.

Train is used for inner scaffold cross-validation, QUBO hyperparameter
selection, and final refitting. Validation is evaluated once after every method
has been selected from train. Validation labels may not alter QUBO
coefficients, subset size, aggregation, or the acceptance thresholds. Test is
not present in any development score matrix.

## Matrix Admission Gate

All three seed runs must finish all 960 receptor-ligand pairs without a failed
pair. The primary matrix is the across-seed median; the minimum score is kept as
a sensitivity matrix.

Before enrichment metrics are calculated, the matrix is rejected if it
contains a nonnegative representative Vina score or a receptor-ligand seed
range greater than `2.0 kcal/mol`. Rejection triggers a label-blind search
diagnostic. It does not permit deleting a ligand, deleting a receptor, or
selectively replacing a score cell.

## Method Selection

Four scaffold-group folds are constructed inside train with seed `20261101`.
Every score normalization bound and every QUBO term is fitted only on the
corresponding inner-training folds.

Compared methods are:

- single-best receptor;
- exhaustive train-selected subset;
- greedy train-selected subset;
- all four receptors;
- exact fixed-size random-subset distribution;
- coverage QUBO;
- discriminative QUBO.

Candidate subset sizes are one, two, and three. Multi-receptor aggregation may
use minimum or mean normalized score. Inner selection prioritizes BEDROC20,
then PR-AUC and ROC-AUC.

## Validation Acceptance

The final train-refitted QUBO candidate is compared with the train-selected
single-best receptor on validation. Every check must pass:

| Check | Required |
|---|---:|
| Primary BEDROC20 delta | at least +0.020 |
| Primary ROC-AUC delta | at least 0 |
| Primary PR-AUC delta | at least 0 |
| Minimum-matrix BEDROC20 delta | at least 0 |
| Paired-bootstrap BEDROC20 CI95 lower bound | at least 0 |

The bootstrap uses 5,000 paired resamples and seed `20261102`. A passing result
only authorizes manual review. It does not automatically release test.

## Interpretation

This protocol gives the QUBO hypothesis a fair opportunity while retaining a
strong fail-closed path. If QUBO fails, single-best remains the scientific
baseline and the MAPK14 test stays locked. If QUBO passes, its exact subset,
coefficients, aggregation, solver settings, and metrics must be committed
before a separate one-time test authorization.

Neither outcome can by itself establish quantum advantage or biological
activity.
