# Stage 6 MAPK14 AutoDock Vina-GPU 2.1 Train-160 Preregistration

## Purpose

This experiment asks whether AutoDock Vina-GPU 2.1 can replace the official
AutoDock Vina 1.2.7 CPU engine in a later, separately trained workflow. It is a
consumed-training engine-equivalence and throughput pilot. It does not test
QUBO benefit or generalization.

## Frozen Data Boundary

- 160 consumed development ligands: 80 active and 80 decoy.
- Five previously prepared MAPK14 receptors.
- Three seed replicates: 20260801, 20260802, and 20260803.
- Total: 2,400 receptor-ligand-seed pairs.
- Validation and test rows remain closed.
- Four macrocycles use the audited rigid PDBQT replacements; no Meeko `CG*`
  or `G*` closure pseudoatoms are present.

## Runtime Identity

- Upstream repository: `DeltaGroupNJUPT/Vina-GPU-2.1`.
- Source commit: `180272b8a5265d6ed9664178345933cebe2cd349`.
- Method: AutoDock-Vina-GPU 2.1.
- Build: NVIDIA OpenCL 3.0, `SMALL_BOX`, precompiled binary kernels.
- Search lanes: 8,000.
- Search depth: upstream heuristic.
- RILC-BFGS: enabled.
- Modes: 9; energy range: 3 kcal/mol.
- Box center: `(-0.49, 3.26, 21.83)` Angstrom.
- Box size: `(22, 24, 32)` Angstrom.

The executable, both OpenCL kernels, Makefile, source commit, version probe,
and compile settings are hashed into an immutable runtime lock before the
first real pair is run.

## Seed Policy

The official CPU evidence used `base_seed + seed_offset` for each ligand.
Upstream Vina-GPU virtual-screening mode initializes one RNG and consumes it
across ligands in filesystem iteration order; that order is not explicitly
sorted by the program. Therefore this pilot invokes one ligand per process and
uses the exact corresponding CPU pair seed. This removes directory-order
ambiguity and makes interrupted runs independently resumable.

## Compatibility Gate

Before the 2,400-pair run, one real pair must finish with the frozen MAPK14 box:

- receptor `MK14_2BAJ_aligned`;
- ligand `MK14_active_L000005`;
- seed `20260801`.

The gate fails on a missing/nonfinite pose score, absolute score above 100
kcal/mol, OpenCL memory error, protein/pocket/relation/grid limit error, or any
evidence that kernels were compiled from source during the timed run.

## Equivalence Gate

All checks must pass:

- all 2,400 pairs present;
- overall median absolute score delta at most 0.5 kcal/mol;
- overall P95 absolute score delta at most 1.0 kcal/mol;
- minimum receptor-seed Spearman correlation at least 0.95;
- median receptor-seed top-5% overlap at least 0.80;
- active/decoy mean signed-delta gap at most 0.5 kcal/mol;
- throughput at least 5x the recorded 32-vCPU CPU reference.

The CPU reference contains 3,840 pairs in 27,872.613 seconds. Consequently the
2,400 GPU pairs must finish within about 3,484 seconds of summed pair process
wall time to satisfy the 5x throughput gate.

## Decision Boundary

A pass permits only a new preregistration for complete Train-696 recomputation
with the exact locked GPU runtime. Existing CPU and GPU score matrices cannot
be mixed. A failure retains official CPU Vina as the supported engine and does
not weaken the already completed Stage 5 independent validation result.
