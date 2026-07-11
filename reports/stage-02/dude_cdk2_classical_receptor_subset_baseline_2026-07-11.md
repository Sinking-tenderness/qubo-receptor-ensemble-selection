# Classical Receptor Subset Baseline

## Method

Before introducing QUBO, every receptor subset of size 1, 2, and 3 was
enumerated. For each subset, ligand scores were aggregated with either the
minimum score or the mean score. The subset was selected using **train ROC-AUC
only**, then evaluated unchanged on validation and test.

This is a classical exhaustive baseline, not a QUBO result.

## Subsets Selected On Train

| Subset size | Minimum-score method | Mean-score method |
|---:|---|---|
| 1 | 1JVP | 1JVP |
| 2 | 1AQ1 + 1JVP | 1AQ1 + 1JVP |
| 3 | 1AQ1 + 1HCL + 1JVP | 1AQ1 + 1HCL + 1JVP |

Train ROC-AUC for the selected subsets:

| Subset size | Minimum score | Mean score |
|---:|---:|---:|
| 1 | 0.700 | 0.700 |
| 2 | 0.722 | 0.728 |
| 3 | 0.722 | 0.700 |

## Evaluation Of The Train-Selected Subsets

| Subset | Method | Train ROC-AUC | Validation ROC-AUC | Test ROC-AUC |
|---|---|---:|---:|---:|
| 1JVP | min/mean | 0.700 | 0.550 | 0.850 |
| 1AQ1 + 1JVP | min | 0.722 | 0.450 | 0.850 |
| 1AQ1 + 1JVP | mean | 0.728 | 0.500 | 0.850 |
| 1AQ1 + 1HCL + 1JVP | min | 0.722 | 0.450 | 0.850 |
| 1AQ1 + 1HCL + 1JVP | mean | 0.700 | 0.500 | 0.800 |

## Interpretation

The best train score improves slightly when 1AQ1 and 1JVP are combined, but
the improvement does not appear on the tiny validation set. Adding 1HCL does
not improve the train-selected ROC-AUC and the three-receptor mean baseline
is lower on test.

This is not evidence that 1JVP is universally best. Validation and test each
contain only two actives, so the estimates are highly unstable. The useful
result is methodological: a subset must be selected on train and evaluated
unchanged elsewhere. QUBO should be compared against this baseline under the
same split and the same budget.
