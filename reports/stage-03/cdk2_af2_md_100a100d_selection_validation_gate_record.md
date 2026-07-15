# Stage 3 CDK2 AF2/MD 100A/100D Selection Validation Gate

## Scope

This gate used the complete three-seed exhaustiveness-32 aggregate matrices for
eight CDK2 AF2/MD receptor medoids and 200 DUD-E ligands. The median-across-seed
matrix was the primary input and the minimum-across-seed matrix was sensitivity
only.

The scaffold-disjoint split was locked before this analysis: 60 actives and 60
decoys for train, 20 and 20 for validation, and 20 and 20 for test. Receptor
utility and QUBO coefficients were estimated from train. QUBO weights, subset
size, and receptor-score aggregation were selected on validation. No test score
or test metric was computed.

## Input Integrity

- Received aggregate archive size: 58,814 bytes
- Received aggregate archive SHA-256:
  `2BCB71D8E2558DF691131370FB526D07E93CF8798D87F251AE28838E1D0827A1`
- Aggregate long table: 1600 rows, no duplicate receptor-ligand pairs
- Median matrix: 200 ligands x 8 receptors, no missing scores
- Minimum matrix: 200 ligands x 8 receptors, no missing scores
- Median matrix rebuilt from the long table with maximum absolute delta `0.0`
- Minimum matrix rebuilt from the long table with maximum absolute delta `0.0`
- Active/decoy counts: 100/100
- Seed warnings: 61 pairs
- Warning pairs by split: train 37, validation 12, test 12
- Scaffold overlap between splits: none

The matrix, warning table, aggregate summary, split manifest, and split summary
were all verified against fixed SHA-256 values before analysis.

## Selection Protocol

- Per-receptor score normalization: min-max bounds fitted on train only
- Candidate subset sizes: 2 and 3
- Receptor aggregation candidates: normalized minimum and normalized mean
- Train utility: BEDROC alpha=20, normalized across receptors
- Active-coverage fraction: top 10% of train ligands
- QUBO coverage, overlap, and redundancy weights: `0, 0.25, 0.5, 1.0`
- Cardinality penalty: 10.0
- QUBO grid trials: 256
- Exact binary states checked per QUBO: `2^8 = 256`
- Validation selection metric: BEDROC alpha=20
- Validation tie breakers: PR-AUC, then ROC-AUC

The cardinality penalty produced the requested subset size in every trial. The
256 weight/aggregation trials collapsed to only four unique
subset-aggregation candidates.

## Train Signal and Complementarity

| Receptor | Train ROC-AUC | Train PR-AUC | Train BEDROC | Active hits in train top 10% |
|---|---:|---:|---:|---:|
| C00 F001 | 0.734 | 0.735 | 0.794 | 10/60 |
| C01 F005 | 0.708 | 0.733 | 0.897 | 10/60 |
| C02 F031 | 0.639 | 0.659 | 0.791 | 10/60 |
| C03 F041 | 0.588 | 0.618 | 0.749 | 8/60 |
| C04 F066 | 0.601 | 0.596 | 0.633 | 9/60 |
| C05 F074 | 0.640 | 0.681 | 0.869 | 11/60 |
| C06 F077 | 0.711 | 0.778 | 0.958 | 12/60 |
| C07 F095 | 0.726 | 0.772 | 0.935 | 11/60 |

- Pairwise train Spearman range: 0.460 to 0.817
- Lowest-correlation pair: C01-C04
- Highest-correlation pair: C06-C07
- Union of actives appearing in at least one receptor train top 10%: 21/60

The receptor rankings are not identical, but the strongest train receptors C06
and C07 are also the most highly correlated pair. Rank diversity therefore
exists without automatically yielding useful ensemble generalization.

## Validation Results

| Method | Subset | Aggregation | ROC-AUC | PR-AUC | BEDROC | EF5% |
|---|---|---|---:|---:|---:|---:|
| single-best on train | C06 | single | 0.638 | 0.679 | 0.891 | 2.0 |
| greedy size 2 | C01+C06 | mean | 0.665 | 0.682 | 0.846 | 2.0 |
| all 8 | all receptors | mean | 0.605 | 0.637 | 0.776 | 2.0 |
| QUBO validation candidate | C01+C06+C07 | mean | 0.600 | 0.629 | 0.722 | 1.0 |

The QUBO-selected subset used coverage weight 0, overlap weight 0, and
redundancy weight 0. It therefore reduced to selection by normalized single-
receptor train utility. The same C01+C06+C07 mean subset was also obtained by
the size-3 exhaustive-train and greedy baselines.

For uniformly random size-3 mean subsets, validation BEDROC had mean 0.731 and
5th/95th percentiles 0.388/0.957. The QUBO candidate BEDROC of 0.722 was below
that random-subset mean and below the train-selected single receptor.

The primary reason is unstable transfer of receptor utility. C07 had train
BEDROC 0.935 but validation BEDROC 0.299. C00 had lower train BEDROC 0.794 but
validation BEDROC 0.976. A train-only utility ranking cannot know this reversal.

## Minimum-Matrix Sensitivity

The fixed subsets and aggregation rules were transferred to the minimum matrix
without retuning.

| Method | Validation ROC-AUC | Validation PR-AUC | Validation BEDROC | EF5% |
|---|---:|---:|---:|---:|
| single C06 | 0.603 | 0.654 | 0.885 | 2.0 |
| all-8 mean | 0.600 | 0.611 | 0.662 | 1.0 |
| QUBO C01+C06+C07 mean | 0.610 | 0.649 | 0.745 | 1.0 |

The sensitivity matrix does not reverse the main conclusion: the QUBO
candidate remains below the train-selected single receptor on early enrichment.

## Gate Decision

Candidate generation and exact QUBO solution passed technically, but the
candidate did not pass the scientific freezing gate. The locked test set must
remain unopened.

This result does not show that QUBO is generally ineffective. It shows that the
current train-utility plus active-coverage/overlap/redundancy objective did not
generalize across this single scaffold split and that its tested coefficient
range produced very little subset diversity.

Before final test evaluation:

1. include a size-1/null-ensemble option rather than forcing two or three
   receptors;
2. normalize utility, coverage, overlap, and redundancy terms to comparable
   scales;
3. compare the current objective with a discriminative variant that penalizes
   early decoy exposure;
4. evaluate those fixed objective families with scaffold-group cross-validation
   on the 160 development ligands only; and
5. preregister a pass rule before choosing one frozen candidate for the 40
   locked test ligands.

## Integrity

- Validation-gate config SHA-256:
  `0F12A1F53EEF7F322F26912B8C3B1D8DF8835D634B7C07D0925944CA6E7C86DD`
- Analysis implementation SHA-256:
  `1D84DCE7A2F2D07AAE293671B04C21B2C6F7FD4DC15B594B29658463FE3C51A0`
- Receptor metrics SHA-256:
  `14CAE9AFCCEC1A28B588EC545E55C70657C34E6DD643E1A42088A20ED598D92B`
- Baseline comparison SHA-256:
  `391C0A78B69AD7BF6070DE0E2906248AB394F8E94A01ACBE0D6E3CC6CE5DA95C`
- QUBO trial table SHA-256:
  `8EA0BC9ACEED59480778F786DD5BD022B7724EB7C5770F9E8E49603C7B7A053B`
- Candidate protocol SHA-256:
  `8930638FD06670833201EBFDBB1C108980F42B3062EF2E8ED4CDD857B876E032`
- Validation-gate summary SHA-256:
  `7080DC3AA8BE93AE8D9F268EB72869917E7FD4D0DABC017ADA337328D4DE767F`

Generated tables remain ignored run artifacts. The tracked config, script,
tests, and this record are sufficient to regenerate them from the hashed input
matrices.
