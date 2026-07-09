# Stage 2 Batch Docking Record: DUD-E CDK2 Subset Against 1AQ1

## Purpose

This module extends the validated 1AQ1-STU redocking protocol to a small
DUD-E CDK2 active/decoy subset.

The goal is to create a reproducible single-receptor docking score table that
can later be evaluated with ROC-AUC, EF1%, EF5%, BEDROC, and related ranking
metrics.

## Inputs

- Target: CDK2
- Receptor ID: `CDK2_1AQ1_A_prepared`
- Receptor PDBQT: `receptors/prepared/1AQ1_A_receptor.pdbqt`
- Ligand manifest: `data/processed/dude_cdk2_subset_10a_50d_pdbqt_manifest.csv`
- Vina executable: `environment/bin/vina_1.2.7_win.exe`
- Vina version: AutoDock Vina v1.2.7
- Vina config: `configs/1AQ1_STU_redocking_vina.txt`
- Base seed: `20260709`

The Vina box and exhaustiveness were kept consistent with the redocking
baseline:

- center: `(0.52, 27.06, 8.97)`
- size: `(18, 18, 16)`
- exhaustiveness: `16`
- num_modes: `10`

## Command

```powershell
conda run -n qubo-receptor-ensemble python .\scripts\batch_vina_docking.py `
  --manifest .\data\processed\dude_cdk2_subset_10a_50d_pdbqt_manifest.csv `
  --vina-exe .\environment\bin\vina_1.2.7_win.exe `
  --receptor .\receptors\prepared\1AQ1_A_receptor.pdbqt `
  --receptor-id CDK2_1AQ1_A_prepared `
  --config .\configs\1AQ1_STU_redocking_vina.txt `
  --output-dir .\results\docking\dude_cdk2_subset_10a_50d `
  --log-dir .\logs\dude_cdk2_subset_10a_50d `
  --score-table .\results\docking\dude_cdk2_subset_10a_50d_scores.csv `
  --base-seed 20260709 `
  --resume
```

## Outputs

Generated local outputs:

- Score table: `results/docking/dude_cdk2_subset_10a_50d_scores.csv`
- Docked poses: `results/docking/dude_cdk2_subset_10a_50d/`
- Vina logs: `logs/dude_cdk2_subset_10a_50d/`

These generated files are ignored by Git.

## Observed Summary

- Input ligand rows: 60
- Selected ligands: 60
- Successful ligands: 60
- Failed ligands: 0
- Score table rows: 599
- Pose rank 1 rows: 60
- Pose rank 10 rows: 59

One ligand produced only 9 modes within the requested Vina output constraints,
so the score table has 599 rows instead of 600. This is not a docking failure.

Best-pose score summary:

| label | count | mean | std | min | median | max |
|---|---:|---:|---:|---:|---:|---:|
| active | 10 | -9.589 | 1.458 | -12.320 | -9.114 | -7.547 |
| decoy | 50 | -8.767 | 1.000 | -10.980 | -8.821 | -6.661 |

Top 10 ligands by best Vina score:

| rank | ligand_id | label | docking_score | runtime_seconds |
|---:|---|---|---:|---:|
| 1 | CDK2_A0009 | active | -12.32 | 43.490 |
| 2 | CDK2_A0010 | active | -11.06 | 16.006 |
| 3 | CDK2_D0022 | decoy | -10.98 | 43.579 |
| 4 | CDK2_A0003 | active | -10.69 | 49.399 |
| 5 | CDK2_D0037 | decoy | -10.64 | 17.452 |
| 6 | CDK2_D0013 | decoy | -10.41 | 52.158 |
| 7 | CDK2_A0001 | active | -10.33 | 39.075 |
| 8 | CDK2_D0036 | decoy | -10.20 | 5681.034 |
| 9 | CDK2_D0049 | decoy | -10.10 | 29.985 |
| 10 | CDK2_D0016 | decoy | -10.06 | 25.393 |

Runtime note:

- `CDK2_D0036` was a major runtime outlier.
- The same ligand had an earlier RDKit 3D preparation warning:
  `MMFF94_not_converged_code_1`.
- The computer was locked around this part of the run, so this wall-clock
  runtime may include system idle, sleep, or power-management effects.
- This does not prove the docking result is invalid, but it should be inspected
  during failure and false-positive analysis.

## Interpretation

This run establishes a single-receptor docking score baseline for the small
DUD-E CDK2 subset.

The result already shows why virtual screening evaluation is a ranking problem:
several decoys receive stronger Vina scores than many actives. Those cases are
candidate false positives and must not be hidden or removed without a documented
reason.

This run does not yet prove screening quality. The next module must convert the
best-pose score table into ranking metrics such as ROC-AUC, PR-AUC, EF1%, EF5%,
and BEDROC.

## Relevance To Ensemble/QUBO Stage

The output table has the shape needed for future multi-receptor work:

- `target_id`
- `receptor_id`
- `ligand_id`
- `label`
- `pose_rank`
- `docking_score`
- `status`
- `runtime_seconds`
- `seed`
- `software_version`

For the QUBO stage, each receptor conformer will produce an equivalent score
table. Those tables can then be combined into a `ligand x receptor_conformer`
matrix for sparse receptor conformer subset selection.
