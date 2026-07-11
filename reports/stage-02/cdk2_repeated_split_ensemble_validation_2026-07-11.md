# CDK2 Repeated-Split Ensemble Validation

## Purpose

The first MVP used one tiny train/validation/test split. This experiment
repeats the same leakage-aware protocol over five stratified seeds to check
whether the apparent QUBO subset is stable.

## Command

```powershell
conda run -n qubo-receptor-ensemble python scripts/repeat_ensemble_mvp.py `
  --matrix results/metrics/dude_cdk2_three_receptor_score_matrix.csv `
  --ligands data/processed/dude_cdk2_subset_10a_50d_rdkit_qc.csv `
  --receptor CDK2_1AQ1_A_prepared CDK2_1HCL_A_aligned_prepared CDK2_1JVP_P_aligned_prepared `
  --seeds 20260711 20260712 20260713 20260714 20260715 `
  --output results/metrics/repeated_ensemble_mvp.json
```

## Findings

- The train-selected single receptor was `1JVP` in all five repeats.
- QUBO selected `1AQ1 + 1JVP` in four repeats and `1HCL + 1JVP` once.
- QUBO never improved ROC-AUC, PR-AUC, or BEDROC over the selected single
  receptor on the corresponding held-out test split.
- The validation grid selected zero coverage and overlap weights in all five
  repeats; a redundancy weight was nonzero once.

## Interpretation

This is evidence against claiming an ensemble gain from the current three
structures. Their score rankings are too similar, and the benchmark is too
small for the coverage term to identify stable complementary active subsets.
The negative result is useful: QUBO has not yet demonstrated the intended
innovation, and adding more penalty terms to this same matrix would be
overfitting rather than progress.

## Next test

The expanded receptor manifest contains four additional human CDK2 structures.
Each passed a four-ligand pilot with zero docking failures under the separate
fast triage protocol (`exhaustiveness=1`, `num_modes=1`). Structures with
template-excluded residues record the nearest excluded-residue distance to the
co-crystal ligand; all are more than 12.8 Angstrom away. A larger ligand sample
must be docked before these candidates can enter a final QUBO comparison.
