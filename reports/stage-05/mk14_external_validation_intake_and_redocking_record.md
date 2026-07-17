# Stage 5 MAPK14 External-Target Intake and Redocking Record

Date: 2026-07-17

## Purpose

Stage 5 starts an independent target-level evaluation on human MAPK14/p38alpha.
CDK2 Stage 4 development and its locked test are complete; the consumed CDK2
test set must not be reused for method fitting. MAPK14 provides a new target,
a larger DUD-E benchmark, and experimentally observed ATP-site and DFG-out
receptor states.

This record covers target intake, coordinate alignment, structure-only pocket
clustering, receptor preparation, and co-crystal redocking. It does not report
MAPK14 active/decoy screening performance.

## Benchmark Intake

- DUD-E target: `mk14`
- Active rows: 578
- Decoy rows: 35,850
- All 36,428 SMILES parse with RDKit.
- Decoy source IDs: 35,812 unique IDs.
- Duplicate decoy IDs: 37 IDs spanning 75 rows; maximum multiplicity 3.

All decoy rows are retained because repeated source IDs can represent distinct
stereoisomer, protomer, or tautomer records. Future ligand IDs must include the
target, label, and source line number. Rows sharing a source ID must remain in
the same scaffold split to avoid leakage.

The audited input manifest is
`data/stage05_mk14_external_validation_inputs.json` with SHA-256
`41B88984252EFB63A87D5A89EB3E37A6F83C2AF5659B07EA45667A30DFABEB08`.

## Receptor Pool and Alignment

The 2QD9 chain-A coordinate frame is the common reference.

| Structure | Role | Matched C-alpha atoms | Aligned global RMSD (A) |
|---|---|---:|---:|
| 2QD9 | DUD-E DFG-1 reference | reference | 0.000 |
| 1A9U | orthosteric ATP-site state | 330 | 1.406 |
| 3KQ7 | DUD-E secondary DFG-2 state | 346 | 1.716 |
| 3K3J | DFG-out state | 328 | 0.956 |

All Kabsch transforms have determinant approximately +1 and preserve coordinate
record counts. Co-crystal ligands were transformed with the same matrices as
their parent receptors; no independent ligand fitting was performed.

## Pocket Features and Structure-Only Clustering

The pocket definition contains the 29 2QD9 residues within 5 A of LGF. Pocket
C-alpha RMSD values relative to 2QD9 are 2.713 A for 1A9U, 2.428 A for 3KQ7,
and 1.343 A for 3K3J. The 3K3J structure lacks reference pocket residue ALA34;
that missing feature was median-imputed only for the clustering calculation.

A three-cluster Ward baseline produced:

- cluster 0: 1A9U and 3K3J; medoid 1A9U;
- cluster 1: 2QD9 alone;
- cluster 2: 3KQ7 alone.

Only four structures are available, so this clustering is descriptive and
unstable as a selection rule. No structure is removed before redocking or
screening, and the result is not treated as evidence of screening redundancy.

## Receptor and Ligand Preparation

Every receptor uses chain A, excludes water and all HETATM records, and is
prepared with Meeko 0.7.1 using Gasteiger charges. The resulting PDBQT files
contain no HETATM records and share AutoDock atom types
`A,C,HD,N,NA,OA,SA`.

3KQ7 contains equal-occupancy A/B alternatives at LEU74, ARG94, and LEU167.
The deterministic altloc-A rule was applied explicitly as
`A:74=A,A:94=A,A:167=A`; Meeko was not allowed to make an implicit choice.
The acetate residues in 3KQ7 and I46 in 3K3J were removed with the other
non-protein HETATM records.

RCSB ModelServer SDF headers label the ligands as 2D despite nonzero Z
coordinates. RDKit recognizes the conformers as 3D. Explicit hydrogens were
added without moving transformed input atoms, after which Meeko generated the
ligand PDBQT files. SB2, LGF, F4C, and KQ7 have 4, 8, 6, and 9 torsional degrees
of freedom, respectively.

## Common Search Box

The union of all four aligned co-crystal ligand heavy-atom extents was padded
by at least 4 A on every side and rounded upward:

- center: `(-0.49, 3.26, 21.83)` A
- size: `(22, 24, 30)` A

This single box covers both the ATP site and the extended DFG-out region. The
same box, Vina version 1.2.7, exhaustiveness 32, 8 CPU threads, up to 20 modes,
6 kcal/mol energy range, and seed 20260717 are used for every case.

## Redocking Gate

RMSD is symmetry-corrected heavy-atom RMSD in the fixed receptor coordinate
frame. No post-docking rigid-body ligand alignment is allowed.

| Case | Top score (kcal/mol) | Top-1 RMSD (A) | Best RMSD rank | Best RMSD (A) | Top-1 <= 2 A |
|---|---:|---:|---:|---:|---|
| 2QD9-LGF | -10.413 | 1.618 | 14 | 1.201 | yes |
| 1A9U-SB2 | -9.309 | 0.471 | 1 | 0.471 | yes |
| 3K3J-F4C | -10.991 | 0.471 | 1 | 0.471 | yes |
| 3KQ7-KQ7 | -12.749 | 1.186 | 2 | 1.144 | yes |

All four top-ranked poses pass the 2 A reference threshold. For 2QD9-LGF, the
geometrically closest pose is rank 14 even though rank 1 is already acceptable;
this is a concrete reminder that a Vina score is not a geometric truth label or
a binding free energy.

## Decision and Next Gate

The four MAPK14 receptor inputs pass the co-crystal pose-reproduction gate and
may enter a ligand-screening pilot. The next work must:

1. build grouped scaffold train/validation/test splits without duplicate-ID leakage;
2. lock the test split before any receptor or QUBO selection;
3. prepare a small balanced pilot first and verify score parsing/failure handling;
4. expand the development benchmark only after the pilot passes;
5. compare single receptors and classical baselines before fitting a MAPK14 QUBO.

No claim about MAPK14 enrichment, QUBO improvement, quantum advantage, or
cross-target generalization is supported yet.
