# MAPK14 EnOpt-Style XGBoost Fresh Validation Record

## Decision

- Status: `supplementary_xgboost_validation_evaluated_primary_gate_unchanged`
- Authorization: `stage05-mk14-frozen-enopt-xgboost-fresh-validation-20260722-v1`
- Primary gate before evaluation: `fresh_validation_passed_test_locked`
- Model refits: 0
- Hyperparameter changes: 0
- Receptor reselections: 0
- Locked test rows or scores read: 0

The two Train-696 XGBoost models were loaded from their frozen artifacts and
applied once to the admitted fresh-validation score matrices. This analysis
did not alter the primary QUBO acceptance decision.

## Frozen Comparators

`xgboost_all5` used all five validation receptors with shallow trees
(`max_depth=2`, 150 trees, learning rate 0.03).

`xgboost_budget3` used the following three training-selected receptors with
`max_depth=3`, 150 trees, and learning rate 0.03:

- `MK14_2BAJ_aligned`
- `MK14_2QD9_reference`
- `MK14_3MPT_aligned`

The budget-three feature set differs from the QUBO subset only in its final
receptor: XGBoost used 3MPT whereas QUBO used 3KQ7.

## Primary Results

| Frozen method | Receptors | BEDROC (alpha=20) | ROC-AUC | PR-AUC | EF1% | EF5% | EF10% | Actives in top 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pair-synergy QUBO | 3 | 0.5509 | 0.8360 | 0.4265 | 18.39 | 8.78 | 5.32 | 9 |
| XGBoost all-five | 5 | 0.5361 | 0.8376 | 0.3868 | 15.76 | 7.98 | 5.72 | 7 |
| XGBoost budget-three | 3 | 0.5106 | 0.8352 | 0.3643 | 15.76 | 7.71 | 5.45 | 8 |

The all-five tree model had a slightly higher ROC-AUC and EF10%, but the QUBO
subset had higher BEDROC, PR-AUC, EF1%, EF5%, and top-10 active count while
using two fewer receptor columns. Against the receptor-budget-matched tree
model, QUBO had higher BEDROC and stronger earliest enrichment.

Primary BEDROC differences were:

- QUBO minus XGBoost all-five: +0.0147
- QUBO minus XGBoost budget-three: +0.0402

These are preregistered descriptive comparisons. No new significance claim is
made because the supplementary protocol did not specify a paired bootstrap
acceptance endpoint for XGBoost.

## Seed Robustness

| Method | seed0 | seed1 | seed2 | Mean seed BEDROC | Worst seed BEDROC | Sensitivity BEDROC |
|---|---:|---:|---:|---:|---:|---:|
| Pair-synergy QUBO | 0.4872 | 0.5497 | 0.5409 | 0.5260 | 0.4872 | 0.5580 |
| XGBoost all-five | 0.4391 | 0.5391 | 0.5299 | 0.5027 | 0.4391 | 0.5429 |
| XGBoost budget-three | 0.3072 | 0.5092 | 0.4951 | 0.4371 | 0.3072 | 0.5228 |

QUBO exceeded XGBoost all-five by +0.0232 for mean-seed BEDROC and +0.0481
for worst-seed BEDROC. It exceeded XGBoost budget-three by +0.0888 and +0.1801
on the same summaries. The budget-three model was particularly unstable on
seed0, where its BEDROC fell to 0.3072 and only one active appeared in the top
ten ranks.

## Interpretation Boundary

The supplementary result strengthens the evidence that the sparse frozen
three-receptor subset is competitive with a nonlinear classical score-matrix
model on new MAPK14 validation ligands. It also indicates that the QUBO subset
is less sensitive to the individual Vina seed than either frozen XGBoost
model in this panel.

This does not establish QUBO-over-greedy performance because QUBO and the
frozen greedy method selected the same subset. XGBoost and QUBO also solve
different modeling tasks: XGBoost learns a nonlinear activity classifier,
whereas QUBO selects a sparse receptor subset followed by a frozen min-score
rule. The result does not demonstrate quantum execution, quantum advantage,
binding affinity, biological activity, cross-target generalization, or test
performance.

## Recorded Outputs

- Supplementary result: `data/stage05_mk14_enopt_xgboost_fresh_validation_result.json`
- Supplementary result SHA-256: `0063FC3D22A512CE5CD031E8B6491A1F50C5D4D83647DA988935D833F87A76DC`
- Predictions SHA-256: `371B9AFF454227716C4A78A4CF7423AC99E100F7FDDBFA8089D410A561FE1EA8`
- Metrics SHA-256: `8F523A0205BB6C7223A73FABEB4AA0ED939AD47681A39188C7390FF77C3F47EA`
- Frozen model artifact SHA-256: `31A08A422ED88437C28720E2AFD56AD69152D148D55D718273F8EECD16FA04D1`

The locked test remains unavailable. Any test release or QUBO-versus-greedy
confirmatory experiment requires a separate prospective authorization.
