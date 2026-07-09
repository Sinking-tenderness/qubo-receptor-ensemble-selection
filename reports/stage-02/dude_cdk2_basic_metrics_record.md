# Stage 2 Basic Metrics Record: DUD-E CDK2 Subset

## Purpose

This module converts Vina docking scores into a ligand ranking and basic virtual
screening metrics.

The main teaching point is score direction: Vina docking scores are usually
better when they are more negative, while most ranking metrics assume higher
scores are better. The script therefore uses:

```text
ranking_score = -docking_score
```

## Script

```text
scripts/evaluate_virtual_screening.py
```

The script:

- selects `pose_rank = 1` for each successfully docked ligand
- converts `active` to binary label `1`
- converts `decoy` or `inactive` to binary label `0`
- sorts by `ranking_score` from high to low
- computes pairwise ROC-AUC without relying on scikit-learn
- computes enrichment factors for requested top fractions

## Toy Example Validation

The 10-ligand hand-calculation example was used to validate EF calculation.

Observed toy results:

- ligand count: 10
- active count: 4
- ROC-AUC: 0.625
- EF30%:
  - top_n: 3
  - top_active: 2
  - top active fraction: 0.6667
  - overall active fraction: 0.4
  - EF: 1.6667

This matches the manual calculation:

```text
EF30% = (2 / 3) / (4 / 10) = 1.6667
```

## CDK2 1AQ1 DUD-E Subset Results

Input score table:

```text
results/docking/dude_cdk2_subset_10a_50d_scores.csv
```

Generated local outputs:

- `results/metrics/dude_cdk2_subset_10a_50d_ranking.csv`
- `results/metrics/dude_cdk2_subset_10a_50d_metrics.json`

These generated metric files are ignored by Git.

Observed results:

- ligand count: 60
- active count: 10
- non-active count: 50
- ROC-AUC: 0.644
- EF1%: 6.0
  - top_n: 1
  - top_active: 1
- EF5%: 4.0
  - top_n: 3
  - top_active: 2
- EF10%: 3.0
  - top_n: 6
  - top_active: 3

Top ranked ligands:

| rank | ligand_id | label | docking_score |
|---:|---|---|---:|
| 1 | CDK2_A0009 | active | -12.32 |
| 2 | CDK2_A0010 | active | -11.06 |
| 3 | CDK2_D0022 | decoy | -10.98 |
| 4 | CDK2_A0003 | active | -10.69 |
| 5 | CDK2_D0037 | decoy | -10.64 |

## Interpretation

The single-receptor 1AQ1 baseline shows better-than-random ranking on this small
teaching subset, but the result should not be overinterpreted.

EF1% is especially unstable here because `ceil(60 * 0.01) = 1`, meaning EF1% is
based on only the top-ranked ligand. The next step is to add uncertainty
estimation and more metrics such as PR-AUC and BEDROC.

## Relevance To Ensemble/QUBO Stage

These metrics define how a receptor conformer can be scored as a virtual
screening receptor.

In the multi-conformer stage, each receptor conformer will produce its own
ranking and enrichment metrics. Those values can become part of the objective
or validation criteria for sparse receptor conformer subset selection.
