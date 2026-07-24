# Stage 6 Vina-GPU 2.1 Deterministic Batch Bridge

## Prior Result

The frozen single-pair Vina-GPU 2.1 pilot completed all 2,400 consumed-train
pairs. It passed five of seven gates but failed minimum receptor-seed Spearman
(`0.926 < 0.95`) and throughput (`3.20x < 5x`). Its formal status remains
`gpu_equivalence_gate_failed`.

The speed failure was partly caused by launching one process for every pair.
That design was needed because upstream virtual-screening mode traverses a
directory without explicit sorting and consumes one shared RNG stream across
ligands.

## Patch Scope

This bridge applies two hash-pinned host-code changes to upstream commit
`180272b8a5265d6ed9664178345933cebe2cd349`:

1. sort virtual-screening ligand paths lexicographically;
2. initialize batch ligand `i` with `batch_seed + i`.

OpenCL kernels, scoring, search, refinement, and molecular inputs are not
modified. The original kernel binaries must match the frozen v1 hashes.

## Execution

The 160 manifest-ordered ligands are split into contiguous chunks of eight.
Each staged filename begins with its six-digit global `seed_offset`. A chunk
starts with `batch_seed = base_seed + first_seed_offset`; the patch therefore
assigns every ligand the same seed used by its standalone v1 process.

There are 20 chunks per receptor-seed and 300 chunks total. Before the full
run, the first eight pairs for seed0 and `MK14_2BAJ_aligned` must reproduce both
scores and complete pose-file SHA-256 values exactly.

## Frozen Bridge Gate

All conditions must pass:

- 2,400 complete pairs;
- 2,400 exact score matches against v1;
- 2,400 exact pose SHA-256 matches against v1;
- maximum absolute score delta exactly zero;
- throughput at least 5x the recorded 32-vCPU reference;
- throughput at least 1.5x the single-pair GPU pilot.

## Decision Boundary

A bridge pass proves only execution equivalence and speed. It cannot reverse
the v1 rank-equivalence failure. It permits the next consumed-train diagnostic
on the two unstable receptor-seed groups, but does not permit Train-696,
validation docking, engine mixing, or a QUBO claim.
