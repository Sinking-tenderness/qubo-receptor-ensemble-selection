# Two-Receptor Comparison Record: CDK2 1AQ1 and 1HCL

## Purpose

This module compares the first two receptor conformers using the same 60-ligand
DUD-E subset, ligand preparation, docking box, Vina version, exhaustiveness,
and per-ligand seeds.

The purpose is to distinguish score correlation from receptor complementarity.
It is not a QUBO result and does not prove that either structure is biologically
superior.

## Inputs and Integrity

- Receptor A: `CDK2_1AQ1_A_prepared`
- Receptor B: `CDK2_1HCL_A_aligned_prepared`
- Ligands: 10 actives + 50 decoys
- Rank representative: `pose_rank=1`
- Vina: 1.2.7
- Box: center `(0.52, 27.06, 8.97)`, size `(18, 18, 16)`
- Exhaustiveness: 16
- Shared ligand count: 60
- Missing ligand-receptor scores: 0
- Failed ligand-receptor pairs: 0
- Rank-1 seed mismatches: 0

The score matrix contains 60 rows and two receptor columns. The long
representative table contains 120 ligand-receptor rows.

## Score Relationship

- Spearman correlation: `0.789`
- Pearson correlation: `0.810`
- Mean score difference, HCL minus 1AQ1: `+1.020 kcal/mol`
- Mean active score difference: `+1.452 kcal/mol`
- Mean decoy score difference: `+0.934 kcal/mol`
- HCL produced a more negative score for 1/60 ligands;
  1AQ1 produced a more negative score for 59/60.

Because lower Vina scores are better, HCL is generally less favorable for this
ligand subset. The positive difference is a docking-score observation, not a
direct measurement of binding free energy.

## Single-Receptor Metrics

| metric | 1AQ1 | 1HCL |
| --- | ---: | ---: |
| ROC-AUC | 0.644 | 0.576 |
| PR-AUC / AP | 0.459 | 0.285 |
| BEDROC(alpha=20) | 0.653 | 0.377 |
| EF1% | 6.0 | 0.0 |
| EF5% | 4.0 | 4.0 |
| EF10% | 3.0 | 2.0 |
| Top10 active count | 4 | 3 |

Bootstrap intervals remain wide for both receptors. The difference should not
be presented as a statistically decisive superiority claim.

## Active Coverage and Complementarity

Early active coverage was compared by receptor:

| cutoff | 1AQ1 active IDs | 1HCL active IDs | new active from 1HCL |
| --- | --- | --- | --- |
| Top 1 | A0009 | none | none |
| Top 3 | A0009, A0010 | A0009, A0010 | none |
| Top 6 | A0009, A0010, A0003 | A0009, A0010 | none |
| Top 10 | A0009, A0010, A0003, A0001 | A0009, A0010, A0001 | none |

The Top10 ligand lists overlap by 7/10, but the more important active coverage
result is that HCL contributes no new active molecule within the examined early
cutoffs. It changes ranking order and decoy composition, but it does not show
active complementarity in this small benchmark.

This does not mean HCL is useless. It may become complementary for another
ligand set or another conformer pool, and the current DUD-E subset is too small
to support a general statement.

## Simple Ensemble Baselines

### Minimum-score ensemble

For each ligand:

```text
ensemble_score = min(score_1AQ1, score_1HCL)
```

This asks whether any receptor conformer gives the ligand a strong score. It is
optimistic and can amplify a single-conformer false positive.

Observed result:

- ROC-AUC: `0.644`
- PR-AUC: `0.459`
- BEDROC: `0.653`
- EF1%: `6.0`
- EF5%: `4.0`
- EF10%: `3.0`

It is effectively the 1AQ1 result because 1AQ1 gives the lower score for 59/60
ligands.

### Mean-score ensemble

For each ligand:

```text
ensemble_score = (score_1AQ1 + score_1HCL) / 2
```

This asks whether a ligand is reasonably compatible with both conformers. It is
more conservative, but can dilute a ligand that genuinely prefers one state.

Observed result:

- ROC-AUC: `0.630`
- PR-AUC: `0.403`
- BEDROC: `0.555`
- EF1%: `6.0`
- EF5%: `4.0`
- EF10%: `2.0`

Neither aggregation rule demonstrates an ensemble improvement here.

## Runtime Caveat

The initial 1HCL batch process stopped after 56 ligands and was resumed. All
scores, poses, logs, seeds, and status values are complete, but
`runtime_seconds` is blank for 56 resumed rank-1 rows because the original
batch script did not persist a per-ligand checkpoint runtime. Runtime comparison
is therefore excluded from this biological interpretation.

## Interpretation

The two receptors provide correlated but non-identical docking signals. In this
small benchmark, HCL lowers early enrichment and does not add new early active
coverage. The result is useful as a negative/contrast conformer for future
selection experiments, not as evidence that apo structures should be excluded
in general.

The next conformer-selection analysis should therefore include more receptor
conformers and compare active coverage, decoy rejection, chemotype diversity,
and redundancy rather than optimizing only minimum or mean docking score.
