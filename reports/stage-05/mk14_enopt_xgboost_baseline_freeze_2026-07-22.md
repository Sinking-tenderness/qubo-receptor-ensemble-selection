# MAPK14 EnOpt-Style XGBoost Baseline Freeze

Date: 2026-07-22

## Purpose

Recent ensemble-docking literature makes a nonlinear score-matrix model a
necessary classical comparator for the receptor-subset QUBO. Two XGBoost
models were therefore fitted before any fresh-validation docking score was
available:

- `xgboost_all5`: all five receptor columns in the frozen validation union;
- `xgboost_budget3`: exactly three of those five columns, selected inside each
  training cross-validation context.

The models use receptor docking scores only. No molecular descriptors,
fresh-validation rows, or test rows entered fitting or model selection.

## Frozen Method

- Input: 696 Train-696 ligands, balanced as 348 active and 348 decoy.
- Score input: three-seed median matrix for fitting and selection.
- Fold boundary: the same deterministic four grouped folds and seed used by
  the final Train-696 QUBO gate.
- Model selection: three inner folds for every outer split.
- Primary selection metric: BEDROC with alpha 20.
- Hyperparameter grid: 16 preregistered deterministic XGBoost settings.
- Feature search: one all-five set or all ten possible five-choose-three sets.
- Sensitivity: three-seed minimum plus seed0, seed1, and seed2 matrices.
- Validation/test access during fitting: zero rows and zero scores.

## Nested OOF Results

| Method | Receptors | Primary BEDROC20 | PR-AUC | ROC-AUC | Mean-seed BEDROC20 | Worst-seed BEDROC20 |
|---|---:|---:|---:|---:|---:|---:|
| Pair-synergy QUBO | 3 | 0.9225 | 0.7658 | 0.7517 | 0.9158 | 0.9014 |
| Nested exhaustive | 3 | 0.9339 | 0.7865 | 0.7740 | 0.9285 | 0.9158 |
| XGBoost all-five | 5 | 0.9312 | 0.7937 | 0.7761 | 0.9110 | 0.8687 |
| XGBoost budget-three | 3 | 0.9038 | 0.7648 | 0.7503 | 0.8724 | 0.8048 |

The all-five tree model slightly exceeded QUBO on primary BEDROC20 by 0.0087
and on PR-AUC and ROC-AUC, but it was less robust to the individual docking
seeds. Its worst-seed BEDROC20 was 0.0326 below QUBO. The receptor-budget-
matched tree model was below QUBO on primary, mean-seed, and worst-seed
BEDROC20.

These are nested Train-696 OOF estimates, not independent validation. The
model classes also differ: XGBoost learns a nonlinear activity classifier,
whereas the QUBO selects a sparse receptor subset whose scores are combined by
the frozen min-score rule.

## Frozen Models

The final all-five model selected shallow trees (`max_depth=2`, 150 trees,
learning rate 0.03). Its gain fractions were highest for 2QD9 and 3MPT.

The final budget-three model selected:

- `MK14_2BAJ_aligned`;
- `MK14_2QD9_reference`;
- `MK14_3MPT_aligned`.

It used `max_depth=3`, 150 trees, and learning rate 0.03. This differs from the
QUBO subset `2BAJ + 2QD9 + 3KQ7`.

The budget-three subset was not stable across the four outer fits. The four
fits selected four distinct combinations among 2BAJ, 2QD9, 3KQ7, and 3MPT;
3K3J was never selected. This instability is a warning against interpreting
tree feature selection as a settled biological receptor choice.

## Reproducibility Audit

The complete fit was run twice. All nine compared outputs had identical
SHA-256 values, including both XGBoost model files, OOF predictions, trial
tables, summary, and frozen artifact.

- Frozen artifact SHA-256:
  `31A08A422ED88437C28720E2AFD56AD69152D148D55D718273F8EECD16FA04D1`
- All-five model SHA-256:
  `5ECBF66A00729CE958A982E991A40AED729DCA0735A21EE9787C7B653D31C647`
- Budget-three model SHA-256:
  `18876BFFEB44B33D544DD0641C43BE804FCBC929266817B6E789B88BC4389A1D`

## Validation Rule

Fresh validation must proceed in this order:

1. complete and admit the official CPU Vina five-receptor matrix;
2. run the original QUBO fresh-validation evaluator and freeze its pass/fail
   result;
3. load the two frozen XGBoost models without fitting or calibration;
4. report primary and seed-robust metrics plus explicit XGBoost/QUBO deltas;
5. keep the locked test unavailable regardless of the supplementary result.

The XGBoost comparison cannot rescue a failed primary gate or invalidate a
passed one. It tests whether nonlinear score use generalizes beyond Train-696;
it does not demonstrate affinity, biological activity, cross-target
generalization, quantum execution, or quantum advantage.
