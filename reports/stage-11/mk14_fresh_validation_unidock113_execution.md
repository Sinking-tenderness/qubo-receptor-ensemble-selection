# Stage 11 MAPK14 Fresh-Validation Uni-Dock Execution

## Purpose

Test one frozen, exploratory exact-QUBO receptor subset against its QUBO
forward-greedy and direct BEDROC-greedy controls on the untouched Uni-Dock
fresh-validation matrix. The run uses only the six receptors in the union of the
four subsets frozen after Stage 10.

## Frozen Workload

- Ligands: 1,576 fresh-validation molecules (75 actives and 1,501 decoys).
- Receptors: 2BAJ, 2QD9, 3BV2, 3ITZ, 3KQ7, and 4AAC.
- Seeds: 20260801, 20260802, and 20260803.
- Batches: 18.
- Docking pairs: 28,368.
- Uni-Dock: 1.1.3, Vina scoring, exhaustiveness 1024, max step 80,
  refine step 5, one output mode, and the frozen MAPK14 common box.
- Input compatibility: exactly 54 validation decoys are regenerated with Meeko
  0.7.1 rigid-macrocycle handling before docking.

## Frozen Comparisons

- Exact pair-synergy QUBO: 2BAJ + 2QD9 + 3BV2.
- QUBO forward greedy: 3BV2 + 3ITZ + 3KQ7.
- Direct robust-BEDROC greedy: 2BAJ + 2QD9 + 3KQ7.
- Full-Train-696 exact QUBO secondary: 2BAJ + 3BV2 + 4AAC.

The primary gate requires the exact pair-synergy candidate to exceed both
controls for primary, mean-seed, and worst-seed BEDROC alpha 20. The paired
split-group bootstrap 95% lower BEDROC-delta bound must also exceed zero for
both comparisons.

## Remote Run

After extracting the input archive and verifying `bundle_manifest.sha256`, use:

```bash
cd /root/autodl-tmp/stage11_mk14_fresh_validation_unidock113_confirmation_v1

nohup env AUTO_POWEROFF=1 \
  bash scripts/experimental/unidock/run_stage11_mk14_fresh_validation_remote.sh \
  > stage11_mk14_fresh_validation.log 2>&1 &

echo $! | tee stage11.pid
tail -f stage11_mk14_fresh_validation.log
```

Set `AUTO_POWEROFF=0` or omit it when automatic shutdown is not desired. The
script creates or reuses the `qubo-unidock-stage08` conda environment, prepares
the 54 rigid macrocycles, audits inputs before starting the GPU, resumes valid
batch checkpoints, performs an independent matrix audit, evaluates only the
frozen candidates, writes core and diagnostic archives, calls `sync`, and then
requests shutdown when `AUTO_POWEROFF=1`.

## Interpretation Boundary

A pass supports external generalization of a QUBO-formulated global receptor
search. The exact-QUBO candidate also matched the classical exhaustive optimum
among 560 size-three subsets during Stage 10, so this experiment cannot establish
QUBO-specific speedup or quantum computational advantage.
