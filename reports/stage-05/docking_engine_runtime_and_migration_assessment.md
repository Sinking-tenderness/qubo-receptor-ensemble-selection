# Docking Engine Runtime and Migration Assessment

Date: 2026-07-22

## Decision Summary

The shortest defensible path to an independent MAPK14 result remains official
AutoDock Vina 1.2.7 on CPU. Only the fresh validation matrix is missing. A GPU
engine can reduce future runtime, but it cannot be inserted only at validation
after the QUBO, normalization bounds, and comparators were fitted on CPU Vina
scores.

The repository cleanup therefore keeps CPU Vina as the supported mainline and
moves the failed Uni-Dock equivalence branch under
`scripts/experimental/unidock/`.

## Observed CPU Throughput

The projection uses the three completed MAPK14 Train-536 e32 runs, each with
4,288 receptor-ligand jobs, 32 concurrent Vina processes, two CPU threads per
process, and a nominal 64-vCPU allocation.

| Seed run | Wall time | vCPU-seconds per pair |
|---|---:|---:|
| seed0 | 6.74 h | 361.9 |
| seed1 | 12.41 h | 666.6 |
| seed2 | 11.48 h | 616.8 |

The 1.84-fold range is real workload and instance variation. The median 616.8
vCPU-seconds per receptor-ligand-seed job is used for planning; the minimum and
maximum are retained as bounds.

## Remaining CPU Time

Fresh validation contains 1,576 ligands, five receptors, and three seeds, or
23,640 independent Vina jobs.

| Allocation | Observed-range projection | Median projection |
|---|---:|---:|
| 16 vCPU | 6.2-11.4 days | 10.6 days |
| 32 vCPU | 3.1-5.7 days | 5.3 days |
| 64 vCPU | 1.6-2.9 days | 2.6 days |

Allowing for setup, stragglers, transfer, and admission checks, practical
planning remains approximately 2-3 days on 64 vCPU, 5-6 days on 32 vCPU, or
10-12 days on 16 vCPU. The three seeds may be assigned to separate instances;
one 64-vCPU machine is convenient but not scientifically required. Execution
amendment 02 selects the available single 32-vCPU instance with 16 concurrent
two-thread Vina processes.

## From-Scratch CPU Cost

The final MAPK14 evidence path, excluding superseded pilots, contains:

- Train-696: 696 x 8 x 3 = 16,704 jobs;
- fresh validation: 1,576 x 5 x 3 = 23,640 jobs;
- total: 40,344 e32 jobs.

The median projection is 4.5 days on 64 vCPU, 9.0 days on 32 vCPU, or 18.0
days on 16 vCPU. A practical from-scratch allowance is 5-6, 9-11, or 18-22
days, respectively. Reproducing every superseded CDK2/MAPK14 pilot, e16 run,
warning diagnostic, and failed search would add work but is not required to
reproduce the final scientific evidence.

## Historical Full Replay

For budgeting only, the major CDK2 and MAPK14 docking campaigns plus the
planned fresh validation total approximately 52,344 e32-equivalent jobs. This
uses a rough one-half weight for e16 jobs and then adds allowance for e64
diagnostics, redocking, interrupted runs, setup, and result admission.

| Allocation | Practical historical replay estimate |
|---|---:|
| 16 vCPU | 28-36 days |
| 32 vCPU | 14-18 days |
| 64 vCPU | 7-9 days |

This is the answer to "repeat every major experiment," not the remaining cost
of the present project. Replaying superseded failures is normally unnecessary;
their committed summaries and hashes are the reproducibility evidence.

## What Survives an Engine Change

The following upstream assets remain valid for Vina-GPU 2.1, Uni-Dock, or
another engine, subject to input compatibility:

- raw structures, receptor alignment, MD trajectories, and structural pool;
- ligand identities, labels, grouped scaffold splits, and locked data roles;
- common box geometry and crystallographic reference poses;
- general metric, nested-CV, baseline, QUBO, bootstrap, and audit code.

The following evidence cannot be mixed across engines and must be regenerated:

- redocking and search-strength calibration for the new engine;
- every production docking score and seed matrix;
- per-receptor normalization bounds;
- nested-CV tuning, QUBO coefficients, selected subsets, and all comparators;
- fresh-validation preregistration and evaluation outputs.

## Migration Scope

| Route | Required reset | Assessment |
|---|---|---|
| Keep official CPU Vina | Run only 23,640 missing validation jobs | Recommended current path |
| Vina-GPU 2.1 | First run a consumed Train-160 equivalence pilot; then rebuild all 40,344 production jobs and refit every method | Closest GPU candidate, but not a drop-in e32 implementation |
| Uni-Dock | Same full rebuild, plus rigid-macrocycle policy and coordinate-warning handling | Current enhanced profile failed 1 of 7 equivalence checks |
| AutoDock-GPU | Rebuild maps, recalibrate AD4 search/scoring, redock, and regenerate every matrix | Different AutoDock4.2.6 engine; largest protocol change |
| GNINA or another learned scorer | New scoring/search protocol, full matrix rebuild, new bias and generalization analysis | Separate research branch, not acceleration-only |

For Vina-GPU 2.1, 15 Train-696 ligands and 54 fresh-validation ligands with
Meeko flexible-macrocycle closure types require an explicit compatibility
policy. Its `thread` and `search_depth` controls do not map directly to official
Vina `exhaustiveness=32`, so search calibration cannot be skipped.

In practical terms, roughly 60-70% of the upstream data engineering,
structural work, and evaluation code survives a GPU-engine migration, but 100%
of the decisive docking evidence and downstream fitted results must be
regenerated. The test partition remains locked in every route.

## Recommended Execution

1. Run the frozen fresh-validation CPU bundle, splitting seeds across smaller
   CPU instances if a single 64-vCPU instance is unavailable.
2. Admit and aggregate all three complete seed matrices locally.
3. Execute the preregistered fresh-validation evaluation once, without QUBO
   retuning.
4. Treat Vina-GPU 2.1 as a parallel next-engine pilot on consumed Train-160;
   use it for a future full rebaseline only if a frozen equivalence gate passes.

Machine-readable calculations are stored in
`data/stage05_docking_engine_runtime_assessment.json`.
