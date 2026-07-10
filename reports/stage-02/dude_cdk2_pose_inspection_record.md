# Stage 2 Pose Inspection Record: CDK2 1AQ1 Top Hits

## Purpose

This module prepares PyMOL inputs for visual inspection of selected top-ranked
active and decoy docking poses.

The goal is to connect score-based virtual screening results back to structure:
whether a ligand pose enters the STU/ATP pocket, whether it overlaps the
co-crystal reference ligand, and whether high-ranked decoys look chemically or
geometrically suspicious.

## Script

```text
scripts/prepare_pose_inspection.py
```

The script:

- reads the per-ligand ranking table
- selects specified ligand IDs
- extracts model 1 from each Vina PDBQT pose file
- converts the extracted pose to a simple PDB file for PyMOL viewing
- writes a PyMOL `.pml` script that loads:
  - raw 1AQ1 receptor
  - co-crystal STU reference ligand
  - STU 5 Å pocket residues
  - selected docked active and decoy poses

## Selected Ligands

| ligand_id | label | rank | docking_score | reason |
|---|---|---:|---:|---|
| CDK2_A0009 | active | 1 | -12.32 | top-ranked active |
| CDK2_A0010 | active | 2 | -11.06 | second-ranked active |
| CDK2_D0022 | decoy | 3 | -10.98 | top-ranked decoy; high cLogP |
| CDK2_D0036 | decoy | 8 | -10.20 | large ligand; high cLogP; prep warning |

## Command

```powershell
conda run -n qubo-receptor-ensemble python .\scripts\prepare_pose_inspection.py `
  --ranking .\results\metrics\dude_cdk2_subset_10a_50d_ranking_v2.csv `
  --receptor-pdb .\receptors\raw\1AQ1.pdb `
  --output-dir .\results\pose_inspection\cdk2_1aq1_top_poses `
  --pymol-script .\results\pose_inspection\cdk2_1aq1_top_pose_inspection.pml `
  --ligand-ids CDK2_A0009 CDK2_A0010 CDK2_D0022 CDK2_D0036
```

Generated local outputs:

- `results/pose_inspection/cdk2_1aq1_top_pose_inspection.pml`
- `results/pose_inspection/cdk2_1aq1_top_poses/*.pdb`

These generated files are ignored by Git.

## PyMOL Encoding Note

On Windows, PyMOL may read `.pml` files using the system GBK codec. If the `.pml`
contains UTF-8 Chinese paths, PyMOL can raise a `UnicodeDecodeError`.

The workaround used in this lesson was to load the script through Python with an
explicit UTF-8 codec, or alternatively to convert the `.pml` to GBK or place the
inspection files in an ASCII-only path.

## Visual Observation Summary

Observed by PyMOL visual inspection:

- `CDK2_A0009` and `CDK2_A0010` both enter the yellow STU reference pocket.
- `CDK2_D0022` and `CDK2_D0036` also enter the pocket, but they show portions
  extending outside the pocket/reference-ligand region.
- `CDK2_D0022` is the more suspicious decoy by visual inspection because it has
  a large extended region and an orientation pattern that differs from the other
  inspected ligands.
- At least one inspected decoy is visibly larger or more extended than the STU
  reference ligand.
- Visual pose inspection alone cannot prove that a decoy is truly inactive.
  It only provides structural hypotheses that need interaction analysis,
  experimental labels, and broader benchmark checks.

## Interpretation

The inspected top-ranked active ligands reproduce the basic expectation that
high-ranking actives should occupy the ATP/STU binding pocket.

The top-ranked decoy `CDK2_D0022` also occupies the pocket, but its extended
geometry suggests a possible score artifact: it may gain favorable Vina contacts
while not necessarily representing a realistic or selective binding mode.

This is a candidate false positive, not a proven false positive.

## Relevance To Ensemble/QUBO Stage

The score matrix used for receptor conformer selection can contain chemically
or geometrically suspicious high-scoring decoys. If QUBO optimizes early
enrichment without structural quality control, it may select receptor conformers
that over-reward these artifacts.

Pose inspection therefore becomes part of the quality-control loop for future
multi-conformer docking and receptor ensemble selection.
