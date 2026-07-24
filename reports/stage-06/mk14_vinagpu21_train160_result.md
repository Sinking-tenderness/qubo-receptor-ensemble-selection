# Stage 6 MAPK14 Vina-GPU 2.1 Train-160 Result

## Integrity and Execution

- Core archive SHA-256:
  `CEF95E2E7F9C884BBF63E9087CF1190FA642B3455D8A5DF938CD00A92C5EA62E`.
- Every internal bundle-manifest entry verified.
- Runtime: RTX 4090, OpenCL 3.0, `SMALL_BOX`, precompiled kernels.
- Upstream source commit:
  `180272b8a5265d6ed9664178345933cebe2cd349`.
- Compatibility smoke passed in 1.884 seconds.
- All 2,400 consumed-train pairs completed without failure.
- Sequential single-pair time: 5,451.583 seconds, or 0.440 pairs/second.

## Frozen Gate

Five checks passed:

- complete pairs: 2,400/2,400;
- median absolute score delta: 0.082 kcal/mol;
- P95 absolute score delta: 0.617 kcal/mol;
- median receptor-seed top-5% overlap: 0.875;
- active/decoy mean signed-delta gap: 0.038 kcal/mol.

Two checks failed:

- minimum receptor-seed Spearman: 0.926, below 0.95;
- speedup over recorded 32-vCPU Vina: 3.20x, below 5x.

The failed Spearman groups were seed1 with `MK14_2QD9_reference` (0.926) and
seed1 with `MK14_3MPT_aligned` (0.941). The formal status is
`gpu_equivalence_gate_failed`.

## Post-Hoc Diagnostic

This diagnostic does not change the gate result. After taking the three-seed
median, overall Spearman was 0.975 and each receptor was between 0.956 and
0.986. This suggests seed-specific search instability rather than a global
score failure.

## Decision

Do not run Train-696 or fresh validation with this protocol. The next bounded
step is an exact-output deterministic-batch bridge to remove process startup
overhead. Only after that bridge passes may the two unstable consumed-train
groups receive a preregistered search-depth diagnostic.
