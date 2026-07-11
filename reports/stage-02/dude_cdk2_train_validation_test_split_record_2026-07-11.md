# DUD-E CDK2 Train/Validation/Test Split

## Purpose

The receptor subset-selection stage must not use the same ligands for
objective design, parameter tuning, and final performance reporting. This
split is therefore created before QUBO formulation.

## Procedure

- Input: the RDKit-QC table for the 60-ligand teaching benchmark.
- Duplicate control: canonical SMILES duplicates are rejected before splitting.
- Stratification: active and decoy labels are split independently.
- Train fraction: 0.60.
- Validation fraction: 0.20.
- Test fraction: 0.20.
- Seed: `20260711`.
- Script: `scripts/split_ligand_benchmark.py`.

## Counts

| Split | Active | Decoy | Total |
|---|---:|---:|---:|
| train | 6 | 30 | 36 |
| validation | 2 | 10 | 12 |
| test | 2 | 10 | 12 |

The split manifest is generated locally at
`data/processed/dude_cdk2_subset_10a_50d_split_manifest.csv`; generated
processed data remain excluded from Git.

## Experimental Rules

1. Use train ligands to estimate receptor quality, active coverage, and
   redundancy terms for a candidate QUBO objective.
2. Use validation ligands to compare subset sizes, penalty weights, and
   classical/QUBO solver settings.
3. Keep test ligands untouched until the protocol is frozen.
4. Report the test result once, together with the number of failed docking
   pairs and all parameter choices.

This split is small, especially the two active molecules in validation and
test. It is a teaching-scale protocol safeguard, not a statistically stable
estimate of prospective performance. A later study should use a larger
benchmark and, where possible, scaffold-aware or time-aware splits.
