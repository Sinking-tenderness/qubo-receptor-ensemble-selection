# Split-Aware Receptor Screening Evaluation

## Purpose

The three-receptor score matrix was joined with the fixed ligand split so
that receptor quality can be inspected separately on train, validation, and
test ligands. The workflow is implemented in
`scripts/evaluate_matrix_by_split.py`.

## ROC-AUC By Split

| Split | Ligands | Actives | 1AQ1 | 1HCL | 1JVP |
|---|---:|---:|---:|---:|---:|
| train | 36 | 6 | 0.694 | 0.611 | 0.700 |
| validation | 12 | 2 | 0.200 | 0.450 | 0.550 |
| test | 12 | 2 | 0.800 | 0.650 | 0.850 |

The complete JSON output is generated locally at
`results/metrics/dude_cdk2_three_receptor_split_metrics.json`.

## Interpretation

The ranking changes substantially between splits. JVP is slightly ahead on
the training set and has the highest test ROC-AUC, but this cannot be treated
as evidence of generalization because the test set contains only two active
molecules. The validation set also produces very different rankings, which
shows why selecting a receptor or QUBO penalty from the test set would be
invalid.

The EF1% and BEDROC values are even more fragile on 12-ligand splits. They
should be reported descriptively here, not used as stable optimization
targets.

## Rule For The Next Module

The next receptor-subset baseline should be fitted using train only. Candidate
subset size and redundancy penalties may be compared on validation. The test
split remains frozen until the subset-selection protocol is finalized.

This split is random and label-stratified, not scaffold-aware. Analogue bias
and chemical-series leakage therefore remain possible limitations.
