# Stage 2 Ligand Preparation Record: DUD-E CDK2 Subset

## Purpose

This module builds a small, traceable active/decoy ligand set for teaching-scale
virtual screening tests against the CDK2 1AQ1 receptor.

The goal is not to prove that the DUD-E decoys are experimentally inactive.
The goal is to create a reproducible input table and Vina-ready ligand files so
that later docking outputs can be traced back to stable ligand IDs.

## Source Data

- Dataset: DUD-E CDK2
- Access date: 2026-07-09
- Raw actives: `data/raw/dude/cdk2/actives_final.ism`
- Raw decoys: `data/raw/dude/cdk2/decoys_final.ism`
- Raw files are local inputs and are not committed to Git.

Observed raw file summary:

- Actives: 474 rows
- Decoys: 27850 rows
- Actives SHA256: `AEBFB8188B01D7E072FC5D14716D3049C603AAA55ADD4DF5F9720DB4F66A9E3D`
- Decoys SHA256: `765E5D2B0366085C3D3389007506A72A8EE25BE6C155915E5CE0615468B3341C`

## Workflow

### 1. Make A Small Subset

Script:

```powershell
conda run -n qubo-receptor-ensemble python .\scripts\make_dude_subset.py `
  --actives .\data\raw\dude\cdk2\actives_final.ism `
  --decoys .\data\raw\dude\cdk2\decoys_final.ism `
  --output .\data\processed\dude_cdk2_subset_10a_50d.csv `
  --n-actives 10 `
  --n-decoys 50 `
  --seed 20260709 `
  --target-id CDK2
```

Expected output:

- 10 actives
- 50 decoys
- 60 unique ligand IDs

### 2. RDKit SMILES QC

Script:

```powershell
conda run -n qubo-receptor-ensemble python .\scripts\check_ligand_smiles.py `
  --input .\data\processed\dude_cdk2_subset_10a_50d.csv `
  --output .\data\processed\dude_cdk2_subset_10a_50d_rdkit_qc.csv `
  --summary .\data\processed\dude_cdk2_subset_10a_50d_rdkit_qc_summary.json
```

Observed summary:

- Input rows: 60
- RDKit parse OK: 60
- Unique canonical SMILES: 60
- Duplicate canonical SMILES: 0
- Multi-fragment molecules: 0
- Charged ligands: 12 decoys
- Heavy atom count range: 18-37
- Molecular weight range: 249.358-513.034

### 3. Generate 3D SDF Files

Script:

```powershell
conda run -n qubo-receptor-ensemble python .\scripts\prepare_ligand_3d_sdf.py `
  --input .\data\processed\dude_cdk2_subset_10a_50d_rdkit_qc.csv `
  --sdf-dir .\ligands\prepared\dude_cdk2_subset_10a_50d\sdf `
  --manifest .\data\processed\dude_cdk2_subset_10a_50d_3d_sdf_manifest.csv `
  --seed 20260709
```

Observed summary:

- OK: 59
- Warning: 1
- Failed: 0
- Warning ligand: `CDK2_D0036`, `MMFF94_not_converged_code_1`

The warning ligand is kept in the manifest instead of silently removed.
Warnings are part of the experimental record and can be revisited during failure
analysis.

### 4. Batch Prepare Vina PDBQT Ligands

Script:

```powershell
conda run -n qubo-receptor-ensemble python .\scripts\batch_prepare_ligand_pdbqt.py `
  --input-manifest .\data\processed\dude_cdk2_subset_10a_50d_3d_sdf_manifest.csv `
  --pdbqt-dir .\ligands\prepared\dude_cdk2_subset_10a_50d\pdbqt `
  --output-manifest .\data\processed\dude_cdk2_subset_10a_50d_pdbqt_manifest.csv `
  --include-warning-sdf
```

Observed summary:

- PDBQT OK: 60
- Failed: 0
- Skipped: 0
- TORSDOF range: 2-13
- PDBQT atom count range: 19-41

## Interpretation

This module verifies that a small DUD-E CDK2 active/decoy subset can be parsed,
audited, converted to 3D ligand structures, and prepared as Vina PDBQT inputs.

This does not mean that all ligands have chemically perfect protonation states,
that every future docking pose will be reasonable, or that the decoys are true
experimental inactives. Those questions must be handled during docking,
interaction inspection, and enrichment analysis.

## Relevance To Ensemble/QUBO Stage

The stable `ligand_id` values and preparation manifests are the future row index
for the `ligand x receptor_conformer` docking score matrix.

Keeping failed and warning rows visible is important because QUBO-based receptor
subset selection can amplify biased or noisy docking signals if the ligand input
pipeline silently drops difficult molecules.
