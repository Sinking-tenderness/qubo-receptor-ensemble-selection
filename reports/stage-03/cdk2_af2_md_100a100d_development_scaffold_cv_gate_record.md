# Stage 3 CDK2 AF2/MD Development Scaffold-CV Gate

## Scope

This gate tested whether a scale-normalized receptor-selection QUBO generalized
within the 160-ligand development set. The original scaffold-disjoint train and
validation partitions were combined for nested four-fold scaffold CV. The 40
final-test ligands remained locked.

The primary input was the three-seed e32 median matrix. The three-seed e32
minimum matrix was evaluated as sensitivity without retuning subsets or
hyperparameters.

## Preregistered Gate

Before execution, the following promotion criteria were fixed relative to the
nested-CV single-best receptor baseline:

1. primary OOF BEDROC delta at least +0.02;
2. primary OOF ROC-AUC delta at least 0;
3. primary OOF PR-AUC delta at least 0;
4. paired-bootstrap BEDROC-delta 95% CI lower bound at least 0; and
5. sensitivity OOF BEDROC delta at least 0.

All criteria had to pass. Manual review would still be required before any test
release.

## Protocol

- Development ligands: 160 (80 active, 80 decoy)
- Locked final-test ligands: 40 (20 active, 20 decoy)
- Scaffold folds: 4
- Fold composition: 20 active and 20 decoy in every fold
- Scaffold overlap across folds: none
- Score normalization: per-receptor min-max fitted inside each training fold
- Candidate subset sizes: 1, 2, and 3
- Aggregation: normalized minimum or normalized mean
- Utility: train BEDROC alpha=20
- Coverage threshold: top 10% of fold-training ligands
- QUBO term scaling: each term family independently min-max normalized
- Cardinality penalty: 20.0
- Bootstrap iterations: 5000

Two QUBO families were preregistered:

- `coverage_qubo`: utility, active coverage, active overlap, and score
  redundancy;
- `discriminative_qubo`: the same terms plus an early-decoy-exposure penalty.

The coverage family contained 111 fixed configurations and the discriminative
family contained 333. Hyperparameters were chosen only by inner folds. Each
chosen configuration was refitted on the corresponding outer-training folds
and evaluated once on the outer fold.

## Primary OOF Results

| Method | ROC-AUC | PR-AUC | BEDROC alpha=20 | EF5% |
|---|---:|---:|---:|---:|
| single-best | **0.686** | **0.754** | **0.963** | 2.0 |
| exhaustive subset | 0.651 | 0.690 | 0.905 | 2.0 |
| greedy subset | 0.662 | 0.702 | 0.906 | 2.0 |
| all receptors | 0.668 | 0.709 | 0.905 | 2.0 |
| coverage QUBO | 0.677 | 0.714 | 0.896 | 2.0 |
| discriminative QUBO | 0.677 | 0.714 | 0.896 | 2.0 |

Both QUBO families produced identical OOF rankings. They were below the
single-best baseline on all three non-saturated ranking metrics.

## Sensitivity OOF Results

| Method | ROC-AUC | PR-AUC | BEDROC alpha=20 | EF5% |
|---|---:|---:|---:|---:|
| single-best | **0.680** | **0.747** | **0.953** | 2.0 |
| exhaustive subset | 0.662 | 0.703 | 0.912 | 2.0 |
| greedy subset | 0.671 | 0.709 | 0.899 | 2.0 |
| all receptors | 0.673 | 0.704 | 0.880 | 2.0 |
| coverage QUBO | 0.676 | 0.720 | 0.908 | 2.0 |
| discriminative QUBO | 0.676 | 0.720 | 0.908 | 2.0 |

The minimum-matrix sensitivity analysis supports the same conclusion.

## Fold Behavior

The single-best rule selected C06 in every outer fold. Coverage and
discriminative QUBOs selected the same subsets:

| Outer fold | QUBO subset | Aggregation | Primary fold BEDROC |
|---:|---|---|---:|
| 0 | C06 | single/minimum | 0.969 |
| 1 | C01+C06+C07 | mean | 0.899 |
| 2 | C06 | single/minimum | 0.926 |
| 3 | C06 | single/minimum | 0.991 |

The QUBO differed from single-best only in fold 1, where the three-receptor
ensemble reduced early enrichment. The decoy-exposure term did not change any
outer-fold subset.

Final four-fold development tuning chose `coverage_qubo`, target size 1, all
QUBO feature weights 0, and receptor C06. This is a mechanical QUBO solution,
but it is equivalent to the single-receptor baseline and contains no
combinatorial complementarity benefit.

## Acceptance Audit

| Criterion | Observed | Required | Pass |
|---|---:|---:|---|
| Primary BEDROC delta | -0.0674 | >= +0.0200 | no |
| Primary ROC-AUC delta | -0.0086 | >= 0 | no |
| Primary PR-AUC delta | -0.0402 | >= 0 | no |
| BEDROC bootstrap CI95 lower | -0.1924 | >= 0 | no |
| Sensitivity BEDROC delta | -0.0443 | >= 0 | no |

The paired-bootstrap mean BEDROC delta was -0.0688 with 95% interval
[-0.1924, 0.0080]. All 5000 bootstrap replicates were valid.

## Independent Output Audit

- OOF rows: 1920
- Matrix-method groups: 12
- Unique ligands per group: 160
- Locked-test ligand overlap: 0
- Maximum BEDROC difference after independent OOF recomputation: 0.0
- Independently recomputed bootstrap values: exact match
- Test scores evaluated: no
- Test metrics computed: no

## Decision

The development gate is rejected and the final test remains locked.

This is not evidence that QUBO is generally unsuitable for receptor selection.
It is evidence that the current eight-medoid pool does not provide a stable
combination advantage over its dominant C06 conformer. Additional coefficient
tuning on the same 160 ligands would increasingly optimize to this development
benchmark without creating new receptor information.

The next scientific step should therefore change the receptor information
content rather than reopen this QUBO grid:

1. expand the conformer pool with additional structurally diverse MD frames
   and aligned experimental CDK2 structures;
2. preregister a label-independent structural-diversity filter before docking;
3. dock the expanded receptors against development ligands first;
4. retain the same nested-CV and locked-test protocol; and
5. use an additional target or external benchmark before making a general QUBO
   claim.

The current work remains a valid engineering MVP: the QUBO is explicit,
scale-normalized, exactly solvable, leakage-controlled, baseline-compared, and
capable of rejecting its own hypothesis.

## Integrity

- Gate config SHA-256:
  `F4FE2B5B4D5D2C1248C627B0398B9C3EE9BB15FE8E06C6E2555BA17CB498637D`
- Normalized QUBO implementation SHA-256:
  `5B9CA96C67D744BCB4C5CC97434B8404F03704053BF8B02516FB45221001FA81`
- CV implementation SHA-256:
  `CD12CD232E21C52390036AB70468F57BF8504F1B3F004D565DCC97B6B2CC1350`
- Fold assignments SHA-256:
  `4FC391610C582CA93E8106A911417747E180E8BFD343FDBBB6445F1418D568D3`
- Outer-fold results SHA-256:
  `B55018CD8BA07A3F0ECAD615DEAD47BB8BDDA1EDE58A7A3A4439A8772B2012CD`
- Method metrics SHA-256:
  `9CA793498F8A5335CD5E4F8D05E0AAE73BACC30ED5EDEAF95CAA2CCFC3562A53`
- OOF scores SHA-256:
  `756FBA5932AECDA561EC80BC05AECD3A10F372B444ABFCE070B067CFB705EC44`
- Final tuning trials SHA-256:
  `F3DF7259C8DC8CBCD56BA61B6E77C4CECC7F6E51999181A67E91BE4F99CDD108`
- Candidate protocol SHA-256:
  `6F6C6C03B4C89617D4823D94BBD96F49F1607C0AE934EB1BF470A2C63F488B53`
- Summary SHA-256:
  `A123CD07D358EFED13212783483BCBAC8F193CE7D299DE7C0FA8BE641D422D45`

Generated CV tables remain ignored run artifacts. The tracked config,
implementations, tests, and this record regenerate them from the hashed input
matrices.
