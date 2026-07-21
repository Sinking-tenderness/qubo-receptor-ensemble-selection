# Stage 5 MAPK14 Train-696 Matrix and QUBO Gate Record

Date: 2026-07-21

## Returned Docking Evidence

The remote result archive has SHA-256
`F52C7321FA2AE9FB5878EB0DD9806EB089157B23657BAFA71DFA4EB374EE813D`.
All three new-ligand e32 seed runs completed 4,288 of 4,288 receptor-ligand
pairs with zero failed pairs. The audited merge contains:

- 696 development-train ligands: 348 active and 348 decoy;
- eight receptor conformers;
- three paired seeds;
- 5,568 aggregated receptor-ligand pairs; and
- 16,704 complete e32 seed cells.

No validation or test row was present. No e32 cell was replaced, and no e64
diagnostic value entered the matrix.

## Retained Search Uncertainty

The median raw seed range was 0.041 kcal/mol. Fifty-four pairs failed the old
minimum-score consensus diagnostic, two raw ranges exceeded 2 kcal/mol, and one
raw range exceeded 5 kcal/mol. The largest case was the decoy
`MK14_decoy_L008329` against `MK14_2BAJ_aligned`: seed0 scored -1.175 kcal/mol,
whereas seed1 and seed2 scored -8.016 and -8.081 kcal/mol.

The two agreeing seeds make the frozen median -8.016 kcal/mol. This isolated
unfavorable search result did not create a nonnegative median. Consistent with
the inherited e64 evidence, all disagreements were retained as uncertainty;
no ligand, seed, or score was removed or selectively rerun.

Status: `matrix_admission_passed_with_retained_seed_uncertainty`.

## Frozen Train-Only Gate

The 696-ligand analysis inherited the complete train-160 protocol without
retuning:

- four outer grouped-scaffold folds and three inner folds;
- 576 coverage/discriminative QUBO candidates;
- subset sizes two and three;
- min-score and mean-score aggregation;
- the same weight grids, tuning order, and acceptance thresholds; and
- matched linear, single-best, nested exhaustive, nested greedy, and all-eight
  comparators.

Before this freeze, archive completeness, uncertainty counts, and descriptive
singleton/all-eight train metrics had been inspected. No QUBO coefficient,
subset, nested-CV result, or matched-linear comparison had been fitted.

## Selected QUBO

The selected candidate was a three-receptor min-score coverage QUBO:

- subset: `2BAJ + 3KQ7 + 3MPT`;
- active-coverage weight: 1.0;
- active-overlap weight: 1.0;
- seed-stability weight: 0.5; and
- decoy-exposure and redundancy weights: 0.0.

All three seed-specific full-train fits selected the same subset, giving mean
pairwise Jaccard 1.0. The objective had 28 non-cardinality quadratic terms with
a nonzero range, and its subset differed from the matched linear subset
`2BAJ + 3K3J + 3KQ7`. The QUBO was therefore structurally non-degenerate and
selection-stable.

## OOF Result

| Method | ROC-AUC | PR-AUC | BEDROC20 | EF5% |
|---|---:|---:|---:|---:|
| QUBO | 0.7301 | 0.7328 | 0.8743 | 1.7714 |
| Matched linear top-k | 0.7112 | 0.7258 | 0.8866 | 1.8286 |
| Single best | 0.6958 | 0.7127 | 0.8944 | 1.9429 |
| Nested exhaustive | 0.7498 | 0.7563 | 0.8948 | 1.8286 |
| Nested greedy | 0.7423 | 0.7593 | 0.9245 | 1.9429 |
| All eight | 0.7132 | 0.6790 | 0.7418 | 1.5429 |

The QUBO improved global ROC-AUC and PR-AUC relative to its matched linear
comparator, but the preregistered primary objective was early enrichment. Its
primary BEDROC20 delta was -0.01237, mean-seed delta was -0.00028, and
worst-seed delta was -0.01321. Seed0 improved by 0.02405, while seed1 and seed2
were worse by 0.01321 and 0.01167.

The selected QUBO ranked at the 75th percentile among all 56 fixed triples on
primary and mean-seed BEDROC, rather than near the optimum. The nested greedy
baseline was materially stronger and stable across all three seeds.

Status: `train_uncertainty_qubo_gate_failed_validation_unavailable`.

## Interpretation and Decision

Increasing the training panel from 160 to 696 ligands reduced sampling
uncertainty but did not rescue the active-overlap quadratic objective. The
quadratic penalty consistently replaced `3K3J` with `3MPT`; that change was
stable but did not improve early enrichment. The limitation is therefore more
consistent with objective misalignment than with QUBO degeneracy, insufficient
ligand count, or docking execution failure.

The current objective is stopped. Validation remains unavailable and test
remains locked. Further tuning of the same weights on these 696 ligands is not
authorized.

One scientifically distinct option remains: preregister a pair-ensemble-utility
quadratic reward calculated strictly inside nested training folds, and require
it to beat both the matched linear and nested greedy comparators. This would
test whether a quadratic term directly aligned with ensemble early enrichment
is useful. It would still be a train-only method-development result and would
not demonstrate quantum advantage by itself.
