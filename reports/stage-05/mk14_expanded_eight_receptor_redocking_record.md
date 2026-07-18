# Stage 5 MAPK14 Eight-Receptor Redocking Record

Date: 2026-07-18

## Purpose

This gate verifies that the final eight MAPK14 receptor inputs can reproduce
their own co-crystal ligand poses under one preparation and docking protocol.
It does not use active/decoy labels or benchmark scores.

## Input Preparation

The four new receptors (`2BAJ`, `4F9W`, `3OCG`, and `3MPT`) were aligned to
2QD9 chain A and prepared with Meeko 0.7.1 using Gasteiger charges. Water and
all non-protein HETATM records were removed. No residue was rebuilt and
`allow_bad_res` was not used.

RCSB ModelServer SDF files retained ligand bond orders and stereochemistry.
The receptor Kabsch transform was applied to each ligand without independent
ligand fitting. Element-matched fixed-frame RMSD between transformed SDF and
the aligned PDB ligand was 0.000479-0.000511 A. Explicit hydrogens were then
added without moving transformed heavy atoms before Meeko ligand preparation.

## Common Box Amendment

The previous box had center `(-0.49, 3.26, 21.83)` A and size
`(22, 24, 30)` A. It left only 3.8575 A between 2BAJ-1PP and the lower z face,
below the preregistered 4 A margin. The gate stopped before any Vina run.

The geometry-only amendment retained the center and x/y dimensions and changed
only `size_z` from 30 to 32 A, a 6.67% volume increase. It also required all
four existing redocking cases to be rerun so all eight receptors use the same
box. The smallest final crystal-pose margin is 4.044 A.

Amendment SHA-256:
`3B3CB03B1B86FCB0323428A046C744C1D7C680E539966996B0F14D5D61336DA7`.

## Vina Protocol

- AutoDock Vina 1.2.7, `vina` scoring
- center: `(-0.49, 3.26, 21.83)` A
- size: `(22, 24, 32)` A
- exhaustiveness: 32
- CPU threads per case: 8
- maximum modes: 20
- energy range: 6 kcal/mol
- seed: 20260717

RMSD is symmetry-corrected heavy-atom RMSD in the fixed receptor coordinate
frame. No post-docking rigid-body ligand alignment is allowed. Every top-ranked
pose must have RMSD at most 2 A.

## Results

| Case | Role | Top score (kcal/mol) | Top-1 RMSD (A) | Best RMSD rank | Best RMSD (A) | Pass |
|---|---|---:|---:|---:|---:|---|
| 2BAJ-1PP | new | -12.052 | 0.448 | 1 | 0.448 | yes |
| 4F9W-LM4 | new | -10.510 | 0.316 | 1 | 0.316 | yes |
| 3OCG-OCG | new | -10.380 | 0.730 | 1 | 0.730 | yes |
| 3MPT-1GK | new | -8.314 | 1.502 | 3 | 0.584 | yes |
| 2QD9-LGF | existing revalidation | -10.415 | 1.582 | 1 | 1.582 | yes |
| 1A9U-SB2 | existing revalidation | -9.265 | 0.469 | 2 | 0.421 | yes |
| 3K3J-F4C | existing revalidation | -11.012 | 0.495 | 1 | 0.495 | yes |
| 3KQ7-KQ7 | existing revalidation | -12.856 | 1.137 | 1 | 1.137 | yes |

All 8/8 top-ranked poses pass. The largest top-1 RMSD is 1.582 A. The 3MPT
top-ranked pose still passes, while rank 3 is geometrically closer; this again
shows that Vina score is not a geometric truth label or binding free energy.

## Independent Audit

The audit independently reloaded all eight docked PDBQT files, reconstructed
RDKit poses through Meeko, recalculated every symmetry-corrected RMSD, checked
all source/output hashes, and rechecked the eight box margins. Status is
`independent_expanded_redocking_audit_ok`.

- canonical summary SHA-256: `AF6E22DA4D932A0463E7803946C2EEA166E5734B5A4AE1AA062FC708C2832611`
- result table SHA-256: `EAFC1B2AB345648A01A121BC62212BF10A65F846AD014A10A38AC1900D53C80A`
- independent audit SHA-256: `BB33800DAAE1338F81486A0410315FC07E47D297267464361BF14552E2E7D7DF`

## Decision

The final eight-receptor pool passes the common-protocol co-crystal redocking
gate. The next authorized work is to dock only the frozen development-train
ligands against the four new receptors, reuse the existing four-receptor train
scores, and run the preregistered train-only non-degeneracy gate. New validation
sampling and the locked test remain prohibited.

Passing redocking supports pose reproduction only. It does not establish
screening enrichment, receptor complementarity, affinity accuracy, QUBO
benefit, or quantum advantage.
