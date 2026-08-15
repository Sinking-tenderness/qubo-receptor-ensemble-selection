# Stage 42 BACE1 development input preparation

## Frozen scope

Stage 42 is a new outcome-informed development experiment. It retains exactly
the 34 Stage 41c redocking-qualified receptors and does not reinterpret the
failed Stage 41c gate.

The ligand allocator freezes 133 development actives and 133 development
decoys. It also freezes identities for 75/1501 fresh-validation and 75/1501
locked-test active/decoy panels, but Stage 42b prepares structures only for the
266 development rows.

Decoy rows sharing a source molecule ID, canonical isomeric SMILES, or achiral
Bemis-Murcko scaffold are assigned as one connected group. Any decoy group
colliding with an active identity or scaffold is excluded before deterministic
hash-seeded allocation.

The allocator removes stereochemistry from a molecule copy before Murcko
scaffold extraction and again before serialization. This is an implementation
compatibility fix for RDKit 2026.03.1: side-chain deletion can otherwise leave
an invalid double-bond stereo annotation. It does not alter source molecules,
labels, panel sizes, hash seeds, or the preregistered achiral grouping rule.

## Current execution

Stage 42b creates deterministic ETKDGv3 structures and Meeko 0.7.1 PDBQT files
for all 266 development ligands. Flexible macrocycles are attempted first; a
ligand is rerun with rigid macrocycles if Meeko fails or closure pseudoatoms
are detected. The final manifest requires zero failed ligands and zero closure
pseudoatoms.

This stage runs no docking job and does not require GPU compute. It is safe to
reuse the existing `qubo-unidock-stage08` environment.

## Next gate

Only after all 266 ligand inputs pass an independent audit may Stage 42c run
the frozen 34 by 266 by three-seed Uni-Dock development matrix, totaling
27,132 receptor-ligand-seed pairs.
