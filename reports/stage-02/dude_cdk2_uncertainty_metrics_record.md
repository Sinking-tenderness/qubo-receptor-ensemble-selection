# Stage 2 Uncertainty Metrics Record: DUD-E CDK2 Subset

## Purpose

This module extends the basic virtual screening metrics with:

- PR-AUC as average precision
- BEDROC for early recognition
- bootstrap 95% confidence intervals

The goal is to avoid overinterpreting a single metric value from a small
teaching-scale active/decoy set.

## Added Metrics

### PR-AUC / Average Precision

Average precision summarizes how precise the ranking is at the positions where
actives appear.

For each active in the ranked list:

```text
precision_at_rank = actives_seen_so_far / rank
```

Average precision is the mean of these precision values across all actives.

### BEDROC

BEDROC is an early-recognition metric derived from exponentially weighted active
rank positions.

In this implementation:

- alpha = 20
- active molecules near the top receive much larger weight
- the score is normalized against finite-rank best and worst placements
- 0 means the worst active placement
- 1 means the best active placement

### Bootstrap Confidence Interval

Bootstrap resampling repeatedly samples ligands with replacement and recomputes
the metrics.

This gives an approximate uncertainty interval for the metric under the current
small dataset.

## Validation Example

The 10-ligand toy ranking was used again.

Observed values:

- ROC-AUC: 0.625
- Average precision: 0.6528
- BEDROC(alpha=20): 0.8808
- EF30%: 1.6667

Average precision was checked from active ranks 1, 3, 6, and 9:

```text
AP = (1/1 + 2/3 + 3/6 + 4/9) / 4 = 0.6528
```

## CDK2 1AQ1 DUD-E Subset Results

Command:

```powershell
conda run -n qubo-receptor-ensemble python .\scripts\evaluate_virtual_screening.py `
  --score-table .\results\docking\dude_cdk2_subset_10a_50d_scores.csv `
  --ranking-output .\results\metrics\dude_cdk2_subset_10a_50d_ranking_v2.csv `
  --metrics-output .\results\metrics\dude_cdk2_subset_10a_50d_metrics_v2.json `
  --top-fractions 0.01 0.05 0.1 `
  --bedroc-alpha 20 `
  --bootstrap-iterations 1000 `
  --bootstrap-seed 20260710
```

Observed point estimates:

| metric | value |
|---|---:|
| ROC-AUC | 0.644 |
| PR-AUC / average precision | 0.459 |
| BEDROC(alpha=20) | 0.653 |
| EF1% | 6.0 |
| EF5% | 4.0 |
| EF10% | 3.0 |

Bootstrap 95% intervals:

| metric | mean | 95% CI low | 95% CI high |
|---|---:|---:|---:|
| ROC-AUC | 0.645 | 0.411 | 0.840 |
| PR-AUC / average precision | 0.460 | 0.134 | 0.748 |
| BEDROC(alpha=20) | 0.611 | 0.056 | 0.920 |
| EF1% | 5.821 | 0.000 | 12.000 |
| EF5% | 4.490 | 0.000 | 8.571 |
| EF10% | 3.307 | 0.714 | 6.250 |

## Interpretation

The point estimates suggest that the 1AQ1 single-receptor baseline has useful
early recognition on this small subset. However, the bootstrap intervals are
wide, especially for EF1% and BEDROC.

This means the result is useful as a teaching baseline, but it is not enough to
claim a strong screening method. More ligands, independent splits, and
multi-conformer comparisons are needed before drawing research-level
conclusions.

## Relevance To Ensemble/QUBO Stage

QUBO receptor conformer selection should not optimize a single noisy metric on a
tiny set.

These uncertainty estimates are a reminder that future conformer selection must
compare against baselines such as random, clustering, greedy, single-best, and
all-conformer ensemble methods, ideally with train/validation/test splits or
bootstrap uncertainty.
