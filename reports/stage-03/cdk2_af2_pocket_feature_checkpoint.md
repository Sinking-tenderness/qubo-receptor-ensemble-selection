# CDK2 AF2 and Pocket-Feature Checkpoint

## Purpose

This checkpoint implements the first Stage 3 entry point after the Stage 2
score-only QUBO negative result: an official AF2 structure source, a shared
coordinate frame, auditable pocket-geometry features, and a technical docking
smoke test. No virtual-screening or QUBO performance claim is made here.

## Official source and model audit

- Target: human CDK2, UniProt `P24941`.
- Official AlphaFold DB entry: `AF-P24941-F1`, model version 6.
- Model created date supplied by AlphaFold DB: 2025-08-01.
- Downloaded raw model: `receptors/raw/alphafold/AF-P24941-F1-model_v6.pdb`
  (kept local and ignored by Git).
- Raw SHA-256: `9037CF07EADCBAD383095070D147C3961B0EDC24845F115A33F4A7DB185C1211`.
- Audit: 2,398 atoms, 298 residues, chain A, mean pLDDT 88.546, range
  40.31-98.88.

The exact API response fields, URLs, and audit are retained in
`data/stage03_cdk2_alphafold_manifest.json`.

Source accessed 2026-07-12:

- https://alphafold.ebi.ac.uk/api/prediction/P24941
- https://alphafold.ebi.ac.uk/entry/P24941

pLDDT is a model-confidence value. It is not evidence that the predicted
structure has a ligand-competent CDK2 pocket.

## Alignment and pocket schema

The AF2 model was aligned to 1AQ1 chain A by sequence-consistent C-alpha
Kabsch alignment before docking-box reuse.

| Alignment quantity | Value |
| --- | ---: |
| Matched C-alpha atoms | 277 |
| RMSD before alignment | 45.916 A |
| RMSD after alignment | 0.944 A |
| Rotation determinant | approximately 1; no reflection |
| Pocket residues | 21 residues within 5 A of 1AQ1 STU |
| AF2 pocket C-alpha RMSD to 1AQ1 | 0.8202 A |
| Reference-pocket residues present in AF2 | 21/21 |

For each conformer, the feature extractor records residue presence,
side-chain atom count, minimum distance to the fixed 1AQ1 STU coordinates,
and side-chain centroid coordinates. Since all conformers are in the same
frame, these values describe pocket geometry differences. They are not
protein-ligand interaction energies and must not be interpreted as affinity.

## Initial structure-only clustering baseline

The initial pool contains eight crystal conformers plus AF2. Pocket features
form a 105-column matrix; 84 continuous geometry columns are median-imputed,
standardized, and Ward-clustered. The exploratory fixed three-cluster medoids
are 1H00, 1AQ1, and 2C69.

AF2 is assigned to the cluster containing 1H00, 1JVP, and 3RKB, but it is not
that cluster's medoid. This merely establishes a label-free structural baseline
for later comparison with a train-only selection objective. It says nothing
about which conformer ranks actives early.

## AF2 receptor preparation and smoke docking

The aligned AF2 PDB was prepared with the existing ProDy + Meeko procedure:

- Protein residues retained: 298.
- PDBQT coordinate records: 2,912; hydrogen-like atoms: 514.
- HETATM records: 0.
- Gasteiger charge range: -0.549 to 0.345.
- Vina-compatible AutoDock atom types: A, C, HD, N, NA, OA, SA.

A two-active/two-decoy technical smoke run used the Stage 2 common box
`(0.52, 27.06, 8.97)` with size `(18, 18, 16)`, Vina 1.2.7,
`exhaustiveness=1`, `num_modes=1`, four parallel one-CPU workers, and
deterministic per-ligand seeds. All four dockings succeeded, produced PDBQT
poses and logs, and had a parsed rank-1 score:

| Ligand | Label | Rank-1 Vina score (kcal/mol) |
| --- | --- | ---: |
| CDK2_A0029 | active | -9.739 |
| CDK2_A0071 | active | -8.306 |
| CDK2_D0004 | decoy | -8.005 |
| CDK2_D0026 | decoy | -7.547 |

The four-ligand sample is deliberately not used to calculate ROC-AUC, EF,
BEDROC, or a claim about AF2 screening performance. Its sole conclusion is
that the prepared AF2 receptor and shared Vina interface work end to end.
The full technical record is
`data/stage03_cdk2_af2_smoke_docking.json`; generated poses and logs remain
local and ignored by Git.

## MD environment gate

`environment/stage03_openmm.yml` specifies a separate OpenMM 8.5 environment
with PDBFixer and MDTraj. It is intentionally separate from the docking
environment so an MD package upgrade cannot alter Vina/Meeko/RDKit results.
Before creating this environment and before producing any MD frame, the
following protocol must be recorded and reviewed: protonation treatment,
force field, water model, ionic strength, box padding, minimization,
equilibration phases, production length, frame stride, random seeds, and
replicate count. No arbitrary MD frames have been generated.

Official OpenMM documentation accessed 2026-07-12:

- https://docs.openmm.org/latest/userguide/
- https://docs.openmm.org/latest/userguide/application/02_running_sims.html

## Next scientific gate

Run full AF2 screening only after the docking protocol is chosen for the
intended comparison. Build any new QUBO objective from training-fold data only,
then evaluate against random, clustering-medoid, greedy, single-best, and
all-conformer baselines under scaffold-group outer CV. The Stage 2 test scores
must not be reused to tune feature weights or QUBO coefficients.
