# Stage 3 CDK2 AF2 MD E32 Three-Seed Aggregate Record

## Scope

This experiment completed two additional full exhaustiveness-32 score matrices
and combined them with the existing exhaustiveness-32 matrix. Every one of the
480 receptor-ligand pairs therefore has three independent seeded Vina results.
Aggregation was uniform across all pairs; no warning-only score replacement was
performed.

## Source Replicates

| Replicate | Base seed | Wall time (s) | Representative table SHA-256 |
|---|---:|---:|---|
| seed0 | 20260901 | 6517.004 | `21BF11AF6A0F487BACEB71EE9F8CFD1AC0E642FEC0F57563173A439FDF21F342` |
| seed1 | 20360901 | 6639.600 | `3B6E49C1ECBC86F4EA27C60D4AF78DC7C5767C36CC5E98CA13249706BE7832F9` |
| seed2 | 20460901 | 6701.169 | `7DE3E2C90051B2291CC4E04734A0FA38E6740A31F0C96B83132DA708D199B567` |

Each replicate used the same eight prepared MD receptors, 30 actives, 30
decoys, box, Vina 1.2.7 executable, exhaustiveness 32, and 8-worker x 4-CPU
layout. Only the fixed base seed changed.

## Aggregate Outcome

- Receptors: 8
- Ligands: 60 (30 active, 30 decoy)
- Receptor-ligand pairs: 480
- Seed replicates per pair: 3
- Pairs with at least two negative seeded scores: 480/480
- Pairs with a nonnegative minimum score: 0
- Pairs with fewer than two favorable replicates: 0
- Seed-stability warnings: 17/480 (3.54%)
- Warnings with seed range greater than 1 kcal/mol: 17
- Warnings with minimum-to-median delta greater than 1 kcal/mol: 2

The minimum-score matrix ranged from -11.380 to -4.066 kcal/mol. The median-
score matrix ranged from -11.360 to -2.953 kcal/mol. The three-seed protocol
therefore removed all catastrophic nonnegative aggregate cells without changing
the molecular inputs or docking box.

## Stability Interpretation

Fifteen of the 17 warnings had a large full seed range but a minimum-to-median
delta no greater than 1 kcal/mol. In these cases, one run diverged while the
other two supported a similar favorable score region. The four active source
failures from the single-seed e32 matrix are examples: the two added seeds
agreed closely and recovered negative scores.

Two CDK2_D0016 pairs had minimum-to-median deltas greater than 1 kcal/mol:

| Pair | Three seeded scores | Minimum | Median | Minimum-median delta |
|---|---|---:|---:|---:|
| C00-D0016 | -3.297, -5.544, +3.165 | -5.544 | -3.297 | 2.247 |
| C03-D0016 | +70.780, -4.547, -2.953 | -4.547 | -2.953 | 1.594 |

These cases show why the minimum can be controlled by one unusually favorable
seed even after catastrophic positive failures are removed. The median requires
support from at least two of three independent runs and is less sensitive to
both unfavorable and favorable single-seed outliers.

## Matrix Decision Before Enrichment Analysis

Before calculating any ROC-AUC, PR-AUC, BEDROC, EF, complementarity, or QUBO
result, the median-across-three-seeds matrix is designated as the primary Stage
3 screening matrix. The minimum-across-three-seeds matrix is retained only as a
sensitivity analysis.

This decision is based on label-independent search stability rather than on
which matrix later gives better enrichment. It supersedes the provisional
minimum-primary designation in the aggregation execution config but does not
alter that historical config or its generated outputs.

## Integrity

- Aggregate long table SHA-256:
  `513C6841B4F0AF5A4A1ACBC64360B9C8DAF62286AA03E44F0780C665194044C1`
- Minimum score matrix SHA-256:
  `3A7690185E0E77BD62502E6BA7B590062A7127A4698841B65556EC361093FC5F`
- Median score matrix SHA-256:
  `AB10F7D0C8F766B912D55384DBFAF9EBD2B8C652938EC585188BE47F0B101311`
- Seed warning table SHA-256:
  `60B45E7AB91210139BDD7188B3978EA7F2B9E41E9D7A05103E9DC2F639EB5104`
- Aggregate summary SHA-256:
  `79568A19E04D314CE27279D341377B35CC42F08FD445D18114D79EDEC16BE6E2`

## Next Gate

The next step is an exploratory 60-ligand screening gate comparing the primary
median matrix with the minimum sensitivity matrix. It will measure per-receptor
ranking, all-receptor ensemble baselines, score correlation, top-hit active
coverage, and fixed scaffold-split behavior.

The scaffold split contains 18 active and 18 decoy training ligands, but only 6
active and 6 decoy ligands in each validation and test split. These small splits
are a protocol gate, not final evidence of generalization or QUBO benefit. No
QUBO tuning or 200-ligand expansion decision is made until this gate is
reviewed.
