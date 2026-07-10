# Stage 2 Score Matrix Interface Record

## Purpose

This module defines the first score-matrix interface for future receptor
conformer ensemble docking and QUBO-based sparse conformer selection.

The current dataset contains only one receptor conformer:

```text
CDK2_1AQ1_A_prepared
```

However, the data structure is designed so that additional receptor conformers
can be added as columns.

## Script

```text
scripts/build_score_matrix.py
```

The script converts one or more long docking score tables into:

- a representative long table with one row per ligand-receptor pair
- a wide matrix with one row per ligand and one column per receptor conformer
- a summary JSON file

## Representative Score Rule

The current representative score is:

```text
pose_rank_1
```

This means one score per ligand-receptor pair is taken from the top-ranked Vina
pose. This is consistent with the previous metric calculations in Stage 2.

The script also supports `min_score` for future sensitivity analysis, but these
rules should not be mixed without explicit documentation.

## Command

```powershell
conda run -n qubo-receptor-ensemble python .\scripts\build_score_matrix.py `
  --score-table .\results\docking\dude_cdk2_subset_10a_50d_scores.csv `
  --long-output .\results\metrics\dude_cdk2_subset_10a_50d_receptor_long.csv `
  --matrix-output .\results\metrics\dude_cdk2_subset_10a_50d_score_matrix.csv `
  --summary-output .\results\metrics\dude_cdk2_subset_10a_50d_score_matrix_summary.json `
  --representative pose_rank_1
```

Generated local outputs are ignored by Git.

## Output Schema

Representative long table:

```text
target_id
ligand_id
label
receptor_id
representative_score
representative_method
status
pose_count
best_pose_rank
best_docking_score
ranking_score
```

Wide matrix:

```text
target_id
ligand_id
label
CDK2_1AQ1_A_prepared
```

For future multi-conformer work, the matrix will expand as:

```text
target_id
ligand_id
label
CDK2_1AQ1_A_prepared
CDK2_AF2_conf_001
CDK2_MD_frame_0001
CDK2_MD_frame_0002
...
```

## Observed Summary

- Ligands: 60
- Receptor conformers: 1
- Long representative rows: 60
- Actives: 10
- Decoys: 50
- Failed ligand-receptor pairs: 0
- Missing scores for `CDK2_1AQ1_A_prepared`: 0

Score direction:

```text
lower representative_score is better for Vina
ranking_score = -representative_score is higher-is-better
```

## Relevance To QUBO

In the future QUBO formulation, each receptor conformer can be represented by a
binary variable:

```text
x_i = 1 if receptor conformer i is selected
x_i = 0 otherwise
```

The score matrix provides the ligand-level evidence needed to evaluate each
conformer and combinations of conformers.

Examples:

- single-conformer EF1%, EF5%, BEDROC
- all-conformer min-score baseline
- mean-score or consensus-score baseline
- greedy conformer subset baseline
- clustering-based subset baseline
- QUBO-selected sparse conformer subset

The matrix also makes missing values explicit. This is important because docking
failures should not be silently dropped before conformer selection.

## Interpretation

This module does not add a new biological result. It defines the data interface
between Stage 2 single-receptor docking and the next multi-conformer/QUBO stage.

The main achievement is that the current single-conformer baseline can now be
treated as a one-column prototype of the future ligand-by-conformer score
matrix.
