# Stage 6 Vina-GPU 2.1 Targeted Search-Depth Diagnostic

## Authorization

The deterministic-batch bridge passed every frozen execution gate:

- 2,400/2,400 exact scores and complete pose hashes;
- zero maximum score delta from the single-pair GPU reference;
- 2,311.736 seconds for 2,400 pairs;
- 7.536x throughput over recorded 32-vCPU Vina;
- 2.358x throughput over single-pair Vina-GPU.

The accepted core archive SHA-256 is
`9C9E4C1F20B9B6973A5C517999C1D30939AC41C82C398FB13B2D2BFD4137BC85`.
This pass authorizes the bounded diagnostic below. It does not change the
frozen single-pair status `gpu_equivalence_gate_failed`.

## Target Groups

Only the two receptor-seed groups that failed the original 0.95 Spearman gate
are eligible:

- seed1 / `MK14_2QD9_reference`: 0.925981;
- seed1 / `MK14_3MPT_aligned`: 0.940547.

Each profile contains all 160 consumed Train-160 ligands for both groups. This
is 320 pairs in 40 deterministic chunks of eight. No validation or test row is
permitted. Ligand labels are retained as provenance but are not used for
profile selection, and no enrichment metric is calculated.

## Frozen Depth Ladder

With RILC-BFGS enabled, the pinned upstream source calculates heuristic depth
as the integer truncation of:

`max(1, (0.24 * atom_count + 0.29 * torsdof - 3.41) * 1.5)`

The 160 audited PDBQTs span heuristic depths 2 through 15 with median 9. The
ascending fixed-depth ladder is therefore:

1. 16, which is at least as deep as every heuristic value;
2. 24;
3. 32.

The host-only deterministic patch, executable, precompiled kernels, Makefile,
source commit, search box, 8,000 lanes, RILC-BFGS setting, mode count, energy
range, molecular inputs, and pair seeds remain unchanged.

## Profile Gate

A profile must pass every condition:

- 320 unique complete pairs;
- overall median absolute CPU/GPU score delta at most 0.5 kcal/mol;
- overall P95 absolute CPU/GPU score delta at most 1.0 kcal/mol;
- Spearman at least 0.95 in each target group;
- top-5% overlap at least 0.80 in each target group;
- throughput at least 5x recorded 32-vCPU Vina.

Profiles are evaluated in ascending order. The first complete pass is selected.
If accuracy fails while speed still passes, the next depth is run. If speed
fails, all higher depths are skipped. This stopping rule is frozen before any
new GPU score is generated.

## Decision Boundary

A selected profile authorizes only a separately frozen full Train-160
confirmation across all five receptors and three seeds using that one fixed
depth. Diagnostic cells cannot replace heuristic-depth cells, and protocols
cannot be mixed within a score matrix. This diagnostic cannot authorize
Train-696, validation, test release, or any QUBO or quantum-advantage claim.
