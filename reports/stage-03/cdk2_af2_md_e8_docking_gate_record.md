# Stage 3 CDK2 AF2 MD Exhaustiveness-8 Docking Gate Record

## Scope

This gate followed the exhaustiveness-4 search-stability failure. It tested the
same two actives and two decoys across all eight prepared MD medoid receptors
at AutoDock Vina exhaustiveness 8, giving 32 receptor-ligand pairs.

The run also tested a four-worker, one-CPU-per-worker scheduling strategy. A
separate C03 control then repeated the same four ligands and seeds with one
four-CPU worker to compare scores and throughput.

## Fixed Docking Protocol

- AutoDock Vina: 1.2.7
- Receptors: 8 aligned and identically prepared MD medoids
- Ligands: `CDK2_A0029`, `CDK2_A0071`, `CDK2_D0004`, `CDK2_D0026`
- Box center: (0.52, 27.06, 8.97) A
- Box size: (18, 18, 16) A
- Exhaustiveness: 8
- Output modes: 1
- Gate scheduling: 4 workers, 1 CPU per worker, maximum total CPU 4
- Base seed: `20260821`
- Gate config SHA-256:
  `7F9269038B42F1BAB3918AF0C8838369E6A57DBAE494B67A52A5B0534999A490`

The selected ligand order assigned seeds `20260821` through `20260824` to
A0029, A0071, D0004, and D0026, respectively. Receptor coordinates were the
intended experimental variable.

## Execution Result

- Expected receptor-ligand pairs: 32
- Successful pairs: 32
- Failed pairs: 0
- Search-quality warning pairs: 0
- Score range: -10.030 to -6.870 kcal/mol
- Final status: `ok`

All eight receptors produced complete score tables, poses, logs, and a combined
receptor-by-ligand matrix. No representative score was nonnegative or more
than 5.0 kcal/mol less favorable than the parent AF2 smoke-test score.

## Score Matrix

All values are Vina scores in kcal/mol. Lower values are more favorable under
the Vina scoring convention.

| Ligand | Label | C00 | C01 | C02 | C03 | C04 | C05 | C06 | C07 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0029 | active | -9.390 | -9.058 | -9.529 | -10.030 | -7.598 | -8.290 | -8.466 | -8.735 |
| A0071 | active | -8.019 | -7.882 | -7.808 | -7.858 | -7.498 | -7.973 | -7.947 | -8.363 |
| D0004 | decoy | -7.967 | -7.690 | -7.204 | -8.004 | -7.128 | -7.719 | -7.566 | -8.133 |
| D0026 | decoy | -7.651 | -8.322 | -7.717 | -8.237 | -6.870 | -7.952 | -8.086 | -8.183 |

The previously unstable C05-D0026 pair returned -7.952 kcal/mol. This is
consistent with its exhaustiveness-4 and exhaustiveness-8 robustness runs near
-7.8 kcal/mol and confirms that the exhaustiveness-1 value of +26.950 was a
search failure.

## Gate Runtime

The eight receptor batches ran sequentially. Within each batch, four one-CPU
Vina jobs ran concurrently.

- Sum of eight measured batch wall times: 2,052.035 seconds
- Mean receptor batch time: 256.504 seconds
- Fastest receptor batch: 211.367 seconds
- Slowest receptor batch: 288.938 seconds
- Total measured gate time: approximately 34.2 minutes

## CPU Scheduling Control

C03 was repeated with identical receptor, ligands, seeds, Vina settings, and
box. Only CPU scheduling changed.

| Ligand | Seed | 4 workers x 1 CPU | 1 worker x 4 CPU | Score delta |
|---|---:|---:|---:|---:|
| A0029 | 20260821 | -10.030 | -10.030 | 0.000 |
| A0071 | 20260822 | -7.858 | -7.858 | 0.000 |
| D0004 | 20260823 | -8.004 | -8.004 | 0.000 |
| D0026 | 20260824 | -8.237 | -8.237 | 0.000 |

The parallel one-CPU batch took 211.367 seconds. The four serial four-CPU Vina
runs took 160.104 seconds in total. On this machine, one four-CPU worker was
1.32 times faster for this exact four-ligand batch while preserving every
representative score.

This throughput result is hardware- and workload-specific. It does not imply
that one scheduling layout is universally faster on machines with more CPU
cores or different memory bandwidth.

## Output Integrity

- Gate summary SHA-256:
  `714102864AC147B9B05A09181AEE6EA4534458C37948EDB0A088343710275736`
- Exhaustiveness-8 score matrix SHA-256:
  `48252CBEF110403BF8522AF43047A4F71E04224E25508D0ECDA1693ACF070D10`
- Representative long table SHA-256:
  `CD57463F7DF01CF70A9B20EC1870D27A8876C057C52E37958DDE8CBCD9572043`
- C03 serial four-CPU score table SHA-256:
  `D9BA612EABBB6F86FF54C2CFD506EEF4BD8539CE2070289CC228AA994996096A`

Generated poses, logs, score tables, and receptor files remain ignored local
artifacts under `results/runs/`. Git records the protocols and this summary.

## Decision

Exhaustiveness 8 passes the all-receptor four-ligand search gate and replaces
exhaustiveness 1 and 4 for subsequent MD medoid score matrices. This result
validates execution and search robustness for the gate panel only; four
ligands cannot establish enrichment, receptor complementarity, or QUBO value.

For the current four-CPU Windows machine, the next batch should use one Vina
worker with four CPUs. The benchmark should first expand to the fixed
30-active/30-decoy panel across all eight MD medoids (480 pairs). This staged
matrix is large enough to expose receptor-specific failures and support a
preliminary complementarity analysis without immediately committing to the
full 100-active/100-decoy, 1,600-pair run.

If the 60-ligand stage remains technically clean, the same audited protocol can
then expand to the full 200-ligand benchmark for train/validation/test receptor
subset selection and final QUBO comparisons.
