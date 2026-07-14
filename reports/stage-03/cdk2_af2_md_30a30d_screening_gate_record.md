# Stage 3 CDK2 AF2 MD 30A/30D Screening Gate Record

## Scope

This gate evaluated the primary three-seed median score matrix and the
three-seed minimum score sensitivity matrix for eight CDK2 AF2/MD receptor
medoids and 60 balanced DUD-E ligands. The analysis was performed only after
the median matrix had been designated as primary on label-independent search-
stability grounds.

The gate measures exploratory screening signal and receptor complementarity.
It is not a final train/validation/test result and is not used to claim QUBO or
quantum advantage.

## Input Integrity

- Screening-result archive size: 136,481 bytes
- Screening-result archive SHA-256:
  `6FF986B5F04566404F6B5E2019CC03674241252FAFF9DFE705B88878A22F2864`
- Primary median matrix SHA-256:
  `AB10F7D0C8F766B912D55384DBFAF9EBD2B8C652938EC585188BE47F0B101311`
- Minimum sensitivity matrix SHA-256:
  `3A7690185E0E77BD62502E6BA7B590062A7127A4698841B65556EC361093FC5F`
- Fixed scaffold split manifest SHA-256:
  `8E195024F139628245F7289345E92FC9F92A63A5CAED84D6ECCD5375417F1922`

## Primary Median-Matrix Results

All eight individual receptors ranked actives above decoys better than random
on the complete balanced set. Individual ROC-AUC values ranged from 0.669 to
0.797.

| Method | ROC-AUC | PR-AUC | BEDROC alpha=20 | EF5% | Top-10 actives |
|---|---:|---:|---:|---:|---:|
| C00 F001 | 0.797 | 0.797 | 0.964 | 2.000 | 9 |
| C07 F095 | 0.792 | 0.845 | 0.995 | 2.000 | 10 |
| all-receptor minimum | 0.764 | 0.805 | 0.987 | 2.000 | 10 |
| all-receptor mean | 0.777 | 0.819 | 0.991 | 2.000 | 10 |

C00 had the best full-set ROC-AUC. C07 had the best PR-AUC, BEDROC, and
Top-10 active count. Neither all-receptor aggregation rule exceeded the best
single receptor across these metrics.

## Complementarity

- Pairwise Spearman score correlation range: 0.471 to 0.886
- Mean pairwise Spearman correlation: 0.719
- Lowest-correlation pair: C00-C04, Spearman 0.471
- Union of active ligands appearing in at least one receptor Top 10: 17/30
- Receptor with the lowest raw score most often: C00, 25/60 ligands

The receptors are not redundant copies. However, different Top-10 lists do not
by themselves prove useful ensemble complementarity: their union uses up to 80
ranked positions, and weak receptors can contribute different false positives.
C04 is strongly decorrelated from C00 but has weaker individual screening
metrics. Raw minimum aggregation is also sensitive to receptor-specific score
offsets, as shown by C00 supplying the lowest score for 25 ligands.

## Fixed-Split Gate

The scaffold split contains 18 active and 18 decoy training ligands, 6 active
and 6 decoy validation ligands, and 6 active and 6 decoy test ligands.

Using train BEDROC to select within each fixed subset size:

| Candidate | Train ROC | Validation ROC | Test ROC | Train BEDROC | Validation BEDROC | Test BEDROC |
|---|---:|---:|---:|---:|---:|---:|
| single C07 | 0.759 | 0.833 | 0.917 | 0.996 | 0.999 | 0.995 |
| size-2 minimum, C04+C07 | 0.725 | 0.806 | 0.806 | 0.995 | 0.999 | 0.994 |
| size-2 mean, C04+C07 | 0.762 | 0.750 | 0.889 | 0.996 | 0.994 | 0.999 |
| all-8 minimum | 0.731 | 0.778 | 0.889 | 0.983 | 0.994 | 0.995 |
| all-8 mean | 0.753 | 0.833 | 0.889 | 0.989 | 0.999 | 0.971 |

The size-2 candidates did not improve consistently over C07. One median-matrix
size-4 mean subset reached test ROC-AUC 0.944, but its validation ROC-AUC was
0.806 and its train BEDROC gain over C07 was only 0.0008. This isolated result
is not stable evidence of a subset benefit.

EF5% was saturated at 2.0 for nearly every candidate. In the 12-ligand
validation and test splits, EF5% evaluates only the top-ranked ligand. BEDROC
was also frequently near 1.0. These metrics cannot resolve small method
differences reliably at this sample size.

## Minimum-Matrix Sensitivity

The minimum matrix produced the same main conclusions:

- C00 full-set ROC-AUC: 0.798
- C07 full-set PR-AUC / BEDROC: 0.844 / 0.993
- all-receptor minimum ROC-AUC: 0.760
- all-receptor mean ROC-AUC: 0.770
- Top-10 active union: 17/30

Median and minimum aggregation therefore agree that individual MD conformers
contain screening signal, receptor rankings are not identical, and naive use
of every receptor does not outperform the strongest single conformer.

## Decision

This 60-ligand gate passes the MD conformer pool to larger-scale evaluation:

1. technical docking and three-seed aggregation are complete;
2. every receptor has non-random ranking signal;
3. meaningful rank diversity exists;
4. the median/minimum sensitivity conclusion is consistent; and
5. naive ensemble aggregation leaves a genuine subset-selection question.

It does not pass to a QUBO performance claim. The current validation and test
sets are too small, and the fixed size-2 baselines do not beat C07 consistently.

The ligand IDs in the 30A/30D and 100A/100D manifests are dataset-local
sequential identifiers, not stable molecular identifiers. Comparing
`source_molecule_id` and canonical SMILES found only five shared molecules, all
active, and no shared decoys. The five shared molecules were embedded with
different seeds and their PDBQT files were not byte-identical, although their
atom-count, atom-type, torsion, and charge-summary fields agreed.

The completed 60-ligand scores therefore cannot be inserted into the
100A/100D matrix. The next experiment must dock all 200 ligands against all
eight receptors for each of the three fixed e32 seeds. After constructing the
complete three-seed median matrix, use scaffold-aware outer cross-validation
and compare single-best, all-receptor, random, clustering, greedy/exhaustive,
and any new structural/contact QUBO objective. QUBO coefficients must be
selected on training folds only.
