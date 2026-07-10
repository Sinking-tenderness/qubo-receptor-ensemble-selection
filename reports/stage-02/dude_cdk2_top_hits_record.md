# Stage 2 Top Hits Record: DUD-E CDK2 Subset

## Purpose

This module inspects the top-ranked ligands from the 1AQ1 single-receptor
docking baseline.

The goal is not to declare top-ranked decoys as true false positives. The goal
is to identify which molecules deserve closer structural inspection in PyMOL,
ChimeraX, or PLIP.

## Script

```text
scripts/analyze_top_hits.py
```

The script merges:

- the per-ligand ranking table from `evaluate_virtual_screening.py`
- the ligand QC/preparation manifest from the DUD-E subset workflow

It outputs a top-hit table containing:

- rank
- ligand ID
- label
- docking score
- pose path
- molecular weight
- heavy atom count
- formal charge
- rotatable bonds
- TORSDOF
- HBD/HBA
- cLogP
- preparation status
- inspection flags

## Heuristic Inspection Flags

The flags are not proof of false positive behavior. They are triage hints.

Current flags:

- `high_mw`: molecular weight >= 500
- `large_ligand`: heavy atom count >= 35
- `charged`: formal charge is not 0
- `many_rotatable_bonds`: rotatable bonds >= 8
- `high_torsdof`: PDBQT TORSDOF >= 8
- `high_clogp`: cLogP >= 4
- `prep_warning`: ligand preparation warning or non-converged 3D optimization

## Commands

Top 10:

```powershell
conda run -n qubo-receptor-ensemble python .\scripts\analyze_top_hits.py `
  --ranking .\results\metrics\dude_cdk2_subset_10a_50d_ranking_v2.csv `
  --manifest .\data\processed\dude_cdk2_subset_10a_50d_pdbqt_manifest.csv `
  --top-n 10 `
  --output .\results\metrics\dude_cdk2_subset_10a_50d_top10_analysis.csv `
  --summary .\results\metrics\dude_cdk2_subset_10a_50d_top10_analysis_summary.json
```

Top 20:

```powershell
conda run -n qubo-receptor-ensemble python .\scripts\analyze_top_hits.py `
  --ranking .\results\metrics\dude_cdk2_subset_10a_50d_ranking_v2.csv `
  --manifest .\data\processed\dude_cdk2_subset_10a_50d_pdbqt_manifest.csv `
  --top-n 20 `
  --output .\results\metrics\dude_cdk2_subset_10a_50d_top20_analysis.csv `
  --summary .\results\metrics\dude_cdk2_subset_10a_50d_top20_analysis_summary.json
```

Generated local CSV/JSON files are ignored by Git.

## Top 10 Summary

- Top 10 ligands: 10
- Actives: 4
- Decoys: 6

Top 10 ligand list:

| rank | ligand_id | label | docking_score | selected flags |
|---:|---|---|---:|---|
| 1 | CDK2_A0009 | active | -12.32 | large_ligand |
| 2 | CDK2_A0010 | active | -11.06 | high_clogp |
| 3 | CDK2_D0022 | decoy | -10.98 | high_clogp |
| 4 | CDK2_A0003 | active | -10.69 | high_torsdof; high_clogp |
| 5 | CDK2_D0037 | decoy | -10.64 | none |
| 6 | CDK2_D0013 | decoy | -10.41 | none |
| 7 | CDK2_A0001 | active | -10.33 | high_torsdof |
| 8 | CDK2_D0036 | decoy | -10.20 | large_ligand; high_clogp; prep_warning |
| 9 | CDK2_D0049 | decoy | -10.10 | high_clogp |
| 10 | CDK2_D0016 | decoy | -10.06 | charged |

Top 10 flag counts:

- charged: 1
- high_clogp: 5
- high_torsdof: 2
- large_ligand: 2
- prep_warning: 1

## Top 20 Summary

- Top 20 ligands: 20
- Actives: 4
- Decoys: 16

Top 20 flag counts:

- charged: 5
- high_clogp: 6
- high_mw: 1
- high_torsdof: 6
- large_ligand: 3
- many_rotatable_bonds: 3
- prep_warning: 1

The top 10 contains several actives, but ranks 11-20 are all decoys in this
small subset. This shows why early-enrichment metrics and structural false
positive analysis should be interpreted together.

## Candidate Decoys For Structural Inspection

Recommended first inspection set:

- `CDK2_D0022`: rank 3, high cLogP
- `CDK2_D0036`: rank 8, large ligand, high cLogP, previous preparation warning
- `CDK2_D0049`: rank 9, high cLogP
- `CDK2_D0016`: rank 10, formal charge -1
- `CDK2_D0042`: rank 11, high TORSDOF

These molecules should be compared with top actives such as:

- `CDK2_A0009`
- `CDK2_A0010`
- `CDK2_A0003`
- `CDK2_A0001`

## Interpretation

Several top-ranked decoys have properties that may bias Vina scores, including
high hydrophobicity, larger size, charge, or higher torsional complexity.

This does not prove that these decoys are invalid. DUD-E decoys are not
experimentally confirmed inactives, and some may be plausible binders. The next
step is to inspect poses and interactions before deciding whether a top-ranked
decoy is a chemically suspicious false positive.

## Relevance To Ensemble/QUBO Stage

QUBO conformer selection should not blindly optimize a score matrix if the top
scores are dominated by systematic artifacts such as oversized hydrophobic
decoys.

Top-hit analysis helps define quality-control features that can later be used
to audit each receptor conformer before using its scores in an ensemble
selection objective.
