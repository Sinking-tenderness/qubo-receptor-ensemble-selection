# Stage 3 CDK2 AF2 MD 30A/30D Linux Docking Record

## Scope

This experiment expanded the eight prepared CDK2 AF2 MD medoids to the fixed
30-active/30-decoy benchmark on one 32-core Linux CPU instance. The first
uniform matrix used AutoDock Vina exhaustiveness 8. It was retained as an audit
after automatic quality controls found two catastrophic search failures.

## Linux Environment

- Operating system: Ubuntu 22.04.1 LTS
- Available CPU allocation: 32 logical CPUs
- Host CPU reported by `lscpu`: AMD EPYC 9654
- Available instance memory: 60 GB allocation
- AutoDock Vina: 1.2.7 Linux x86-64
- Vina executable SHA-256:
  `F31F774F723BBA7BBE6E9D1C47577020EEA9A8DA16424284C043D22593570644`
- Repository commit used for the e8 matrix:
  `ee4addf`

The platform listing and container CPU model differed. All throughput decisions
therefore used measured wall time rather than the marketplace model label.

## Input Transfer Audit

- Molecular input archive size: 1,622,332 bytes
- Molecular input archive SHA-256:
  `A794E352B55B45B047761522B109FA9D2B76A90826D8AC6B0E6BF651238B0478`
- Processed-manifest archive size: 93,383 bytes
- Processed-manifest archive SHA-256:
  `DF910D49F55A4CCF502C1B0CE145ECA9B0F9FFAF5A674FBA0C6C0EAC2A5634E0`
- Prepared receptor PDBQT files: 8 of 8 present
- 30A/30D ligand PDBQT files: 60 of 60 present
- 100A/100D ligand PDBQT files staged for later use: 200 of 200 present

Remote archive hashes matched the local source hashes before extraction.

## CPU Layout Benchmark

One C03 receptor and the same seeded set of 16 actives and 16 decoys were run
under three layouts. Every layout used 32 total CPUs, exhaustiveness 8, one
output mode, and the same box.

| Layout | Successful pairs | Wall time (s) | Relative to fastest |
|---|---:|---:|---:|
| 8 workers x 4 CPU | 32/32 | 77.124 | 1.00 |
| 16 workers x 2 CPU | 32/32 | 102.037 | 1.32 |
| 32 workers x 1 CPU | 32/32 | 140.299 | 1.82 |

The maximum paired score difference among layouts was exactly 0.0 kcal/mol.
The formal run therefore used eight concurrent Vina jobs with four CPUs each.

## Exhaustiveness-8 Matrix

- Receptors: 8
- Ligands: 60 (30 active, 30 decoy)
- Expected receptor-ligand pairs: 480
- Successful technical runs: 480
- Failed technical runs: 0
- Measured workflow wall time: 1,042.379 seconds (17.4 minutes)
- Initial status: `ok_with_search_warning`
- Score range: -11.370 to +49.330 kcal/mol
- Search-quality warnings: 2

The warning policy flagged nonnegative scores and scores more than 5.0 kcal/mol
less favorable than the median for the same ligand across eight receptors.

| Receptor-ligand pair | Label | Seed | e8 score | Ligand median | Delta |
|---|---|---:|---:|---:|---:|
| C07-A0015 | active | 20260915 | +4.552 | -8.7255 | +13.2775 |
| C00-D0016 | decoy | 20260946 | +49.330 | -5.1425 | +54.4725 |

These values were preserved in the e8 raw table and matrix. Neither was
silently removed or replaced.

## Search-Intensity Rescue

The two warning pairs were rerun with the same receptor, ligand, box, seed,
CPU allocation, and Vina version. Only exhaustiveness changed.

| Pair | e8 | e16 | e32 | e64 |
|---|---:|---:|---:|---:|
| C07-A0015 | +4.552 | -8.802 | -8.802 | not run |
| C00-D0016 | +49.330 | -2.789 | -3.297 | -3.297 |

C07-A0015 converged by exhaustiveness 16. C00-D0016 continued to improve from
exhaustiveness 16 to 32 and then remained unchanged at 64. This demonstrates
that exhaustiveness 8 produced two search failures and that exhaustiveness 16
was still insufficient for one difficult case.

## Output Integrity

- e8 summary SHA-256:
  `A2EBC9D746E1F3B2AAACACAE50A2DB3907DDBE620A7C18AF0E0FBAA2C08C2D7B`
- e8 score matrix SHA-256:
  `8D3E99FAE4AEA0F651E4ADD78B283078FC966EFC13AD2312A7B6F053BF8BC578`
- e8 representative long table SHA-256:
  `274613EB6DC20CCAED3DDED1732801B7C9A386F11F6C0898E8140DBF99785C27`
- e8 search-warning table SHA-256:
  `70735950E7568EDD8A3D4A1B60257526BECA6FE4764A9E27B193E07A532C6C21`

Generated poses, logs, and matrices remain ignored run artifacts. Protocols,
code, tests, and this report are tracked in Git.

## Decision

The exhaustiveness-8 matrix is excluded from enrichment, receptor
complementarity, and QUBO calculations. Two failures among 480 pairs can alter
ligand ranks and manufacture apparent receptor-specific behavior.

Selective replacement of only the two warning cells is also rejected because
it would mix search protocols inside one score matrix. A new uniform
exhaustiveness-32 matrix is the next formal result. The exhaustiveness-8 matrix
remains a reproducible negative result showing why a four-ligand gate alone was
insufficient to establish global search reliability.
