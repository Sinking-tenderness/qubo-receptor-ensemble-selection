# Stage 2 Pose Contact Record: CDK2 1AQ1 Top Hits

## Purpose

This module adds a simple geometry-based contact analysis for selected docked
poses.

The goal is to move one step beyond visual pose inspection by asking which
protein residues are close to each ligand pose and whether the pose contains
candidate polar, hydrophobic, or clash-like contacts.

This is not a replacement for PLIP or detailed manual interaction annotation.
It is a transparent first-pass screening tool.

## Script

```text
scripts/analyze_pose_contacts.py
```

The script parses:

- receptor atoms from `receptors/raw/1AQ1.pdb`
- selected docked ligand pose PDB files from `results/pose_inspection/`

It reports:

- receptor residues within 4.0 Å of each ligand
- candidate polar contacts
- candidate hydrophobic contacts
- possible heavy-atom clashes
- hinge residue contact candidates

## Heuristic Rules

Current rules:

- receptor-ligand heavy-atom distance <= 4.0 Å: close contact
- N/O/S to N/O/S distance <= 3.5 Å: polar contact candidate
- C to C distance <= 4.0 Å: hydrophobic contact candidate
- heavy-atom distance < 2.0 Å: possible clash

These are geometric heuristics only. A real hydrogen bond also depends on
donor/acceptor assignment, hydrogen positions, angle, protonation state, and
local environment.

## Command

```powershell
conda run -n qubo-receptor-ensemble python .\scripts\analyze_pose_contacts.py `
  --receptor-pdb .\receptors\raw\1AQ1.pdb `
  --ligand CDK2_A0009 .\results\pose_inspection\cdk2_1aq1_top_poses\CDK2_A0009_pose1.pdb `
  --ligand CDK2_A0010 .\results\pose_inspection\cdk2_1aq1_top_poses\CDK2_A0010_pose1.pdb `
  --ligand CDK2_D0022 .\results\pose_inspection\cdk2_1aq1_top_poses\CDK2_D0022_pose1.pdb `
  --ligand CDK2_D0036 .\results\pose_inspection\cdk2_1aq1_top_poses\CDK2_D0036_pose1.pdb `
  --contacts-output .\results\pose_inspection\cdk2_1aq1_top_pose_contacts.csv `
  --summary-output .\results\pose_inspection\cdk2_1aq1_top_pose_contacts_summary.json
```

Generated contact CSV/JSON files are local outputs and are ignored by Git.

## Summary

| ligand_id | label | contact residues | polar candidates | hydrophobic candidates | possible clashes |
|---|---|---:|---:|---:|---:|
| CDK2_A0009 | active | 16 | 3 | 36 | 0 |
| CDK2_A0010 | active | 15 | 2 | 30 | 0 |
| CDK2_D0022 | decoy | 16 | 4 | 29 | 0 |
| CDK2_D0036 | decoy | 16 | 2 | 32 | 0 |

Closest residue patterns:

- `CDK2_A0009`: HIS84, LEU83, PHE80, ILE10, ASP145, GLU81
- `CDK2_A0010`: HIS84, LEU83, ILE10, ALA31, ASP145, GLN131
- `CDK2_D0022`: THR14, GLU12, ASN132, GLN131, PHE80
- `CDK2_D0036`: ASP86, LEU83, PHE82, GLU8, ASP145

Candidate polar contacts:

- `CDK2_A0009`: GLU81, LEU83, HIS84
- `CDK2_A0010`: GLN131, HIS84
- `CDK2_D0022`: GLN131, GLU12, THR14
- `CDK2_D0036`: ILE10, ASP86

## Interpretation

All four inspected poses show pocket contacts and no obvious heavy-atom clash
under the 2.0 Å cutoff.

`CDK2_D0022` is therefore not suspicious because it fails to enter the pocket.
It is suspicious because its pose extends differently from the active poses and
its closest-contact pattern shifts toward THR14, GLU12, ASN132, and GLN131.

This matches the PyMOL visual observation that `CDK2_D0022` has a distinct
extended orientation. It should be treated as a candidate false positive for
closer interaction inspection, not as a proven inactive molecule.

## Relevance To Ensemble/QUBO Stage

The future receptor-conformer score matrix should not be interpreted from Vina
scores alone.

Contact-level summaries can help identify receptor conformers that improve
early enrichment by forming plausible hinge and pocket interactions, versus
conformers that mainly reward large, extended, hydrophobic, or geometrically
odd decoy poses.

These contact features may become quality-control descriptors alongside EF,
BEDROC, and docking score in the multi-conformer selection workflow.
