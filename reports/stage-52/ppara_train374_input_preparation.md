# Stage 52a PPARA Train-374 Input Preparation

Stage52 is a separate post-hoc exploratory branch. It preserves the failed
Stage51 confirmation and freezes all 20 Stage51-passing receptors without
structural max-min compression. The development panel contains 187 actives and
187 decoys; fresh-validation and locked-test manifests remain sealed.

Stage52a performs ligand preparation only. RDKit ETKDGv3 generates deterministic
3D conformers, followed by MMFF94 or UFF minimization. Meeko first prepares a
flexible ligand and retries with rigid macrocycles only after a preparation
failure or closure-pseudoatom detection. Final PDBQT files must contain no
CG*/G* closure pseudoatoms.

Each ligand has an identity-bound checkpoint. `--resume` reuses a result only
when the config identity, source row, SDF hash, PDBQT hash, basic PDBQT audit,
and zero-pseudoatom gate all still match. Interrupted runs therefore prepare
only missing or invalid ligands.

Passing Stage52a authorizes construction of the exploratory 20-receptor by 374
ligand by three-seed Uni-Dock matrix, totaling 22,440 receptor-ligand-seed
scores. It does not authorize validation release or any confirmatory, QUBO,
speedup, or quantum-advantage claim.

## Observed Result

All 374 ligands were prepared successfully in source-manifest order: 187
actives and 187 decoys. The output contains 374 SDF files, 374 PDBQT files, and
374 identity-bound checkpoints. All file hashes, source identities, heavy-atom
counts, PDBQT atom counts, torsional degrees of freedom, and checkpoints passed
independent verification. No closure pseudoatom remained and no geometry was
non-finite or degenerate.

MMFF94 converged within 500 iterations for 346 ligands. The remaining 28 had
finite, non-degenerate coordinates and subsequently passed Meeko and all PDBQT
gates, so the non-convergence code is retained as a non-fatal warning. Six
macrocycles were converted to the preregistered rigid representation after
closure-pseudoatom detection. Stage52b exploratory production is authorized.
