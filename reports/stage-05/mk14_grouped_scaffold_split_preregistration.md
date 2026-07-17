# Stage 5 MAPK14 Grouped Scaffold Split Preregistration

Date: 2026-07-17

## Why This Split Is Needed

The MAPK14 DUD-E benchmark contains 578 active rows and 35,850 decoy rows.
The task is ranking, not ordinary balanced classification, but train,
validation, and test roles are still needed to prevent receptor selection and
QUBO fitting from being evaluated on the same ligands.

A scaffold-only split is insufficient for this source because 37 duplicate
decoy source IDs span 75 rows and can represent related stereoisomer, protomer,
or tautomer records. The split therefore groups the connected components formed
by either a shared non-chiral Murcko scaffold or a shared source molecule ID.

## Lossless Ligand Manifest and QC

- Total rows retained: 36,428
- Actives: 578
- Decoys: 35,850
- RDKit parse success: 36,428/36,428
- Unique canonical SMILES: 36,426
- Duplicate canonical SMILES: 2
- Multi-fragment molecules: 0
- Molecules with nonzero formal charge: 13,692
- Heavy-atom range: 6-45
- Molecular-weight range: 114.188-633.354

Charged molecules are retained. Formal charge is a molecular property and not
a reason for deleting a ligand. The full row-level audit remains in the local
processed table; the tracked summary stores counts and a bounded preview.

## Frozen Split

The deterministic seed is `20260717` and target fractions are 60/20/20.

| Split | Active | Decoy | Total | Role |
|---|---:|---:|---:|---|
| train | 348 | 21,469 | 21,817 | protocol and objective development |
| validation | 115 | 7,190 | 7,305 | hyperparameter and method selection |
| test | 115 | 7,191 | 7,306 | one final frozen-protocol evaluation |

Audit results:

- Murcko scaffold count: 22,427
- Connected split-group count: 22,412
- Largest connected group: 293 rows
- Scaffold leakage: none
- Source-ID leakage: none

The locked split file is
`data/processed/stage05_mk14_grouped_scaffold_split.csv` with SHA-256
`20818CB9088DBFCB1098D57C9CBA1979B109F3CCE0C9A4D60B7D34D3BB64AE71`.

## Test Lock

The MAPK14 test partition is `locked_unreleased`. It must not be docked or used
to calculate metrics until receptor inputs, the score aggregation rule, the
QUBO objective, solver settings, and all reported metrics are frozen. The test
will then be released once for final evaluation.

This lock is procedural as well as cryptographic: DUD-E labels are publicly
available, so scientific validity depends on respecting the declared roles even
though the underlying source is not secret.

## Next Work

The immediate next gate is a small balanced train-only docking pilot across all
four redocking-approved receptors. It is for execution, output parsing, and
failure-handling validation only; its metrics are not evidence of screening
performance. A larger development benchmark will be selected only after that
pilot passes.
