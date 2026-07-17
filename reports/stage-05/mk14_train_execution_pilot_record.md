# Stage 5 MAPK14 Train-Only Execution Pilot

Date: 2026-07-17

## Scope

This pilot tests the mechanical path from the locked train split through ligand
preparation, docking against four redocking-approved MAPK14 receptors, pose
parsing, checkpoint resume, and score-matrix construction.

It contains four active and four decoy ligands and is not a virtual-screening
performance experiment. The validation and test partitions were not used. The
test partition remains `locked_unreleased`.

## Pilot Selection

The selector used seed `20260718` and chose eight different split groups and
eight different scaffolds from train.

- 4 neutral actives
- 3 neutral decoys and 1 charge-`+1` decoy
- 25-40 heavy atoms
- 3-9 rotatable bonds

The pilot therefore exercises a charged ligand path and several relatively
large, flexible molecules. All eight RDKit 3D preparations and all eight Meeko
PDBQT preparations completed successfully.

## Docking Protocol

- Vina 1.2.7
- common box center: `(-0.49, 3.26, 21.83)` A
- common box size: `(22, 24, 30)` A
- exhaustiveness: 8
- modes: 5
- CPU per Vina process: 2
- base seed: 20260718
- representative score: pose rank 1

Exhaustiveness 8 is an execution-pilot setting. It is not yet the frozen
development or final-screening protocol.

## Execution Result

All 32 receptor-ligand pairs completed and all 32 representative scores are
present.

| Receptor | Ligands | Failed pairs | Median ligand runtime (s) | Top-score range (kcal/mol) |
|---|---:|---:|---:|---:|
| 1A9U | 8 | 0 | 161.0 | -11.030 to -5.526 |
| 2QD9 | 8 | 0 | 236.0 | -10.120 to -6.097 |
| 3K3J | 8 | 0 | 208.4 | -12.300 to -5.672 |
| 3KQ7 | 8 | 0 | 149.4 | -11.110 to -6.538 |

The first 2QD9 attempt used four simultaneous Vina processes and three jobs
reported `insufficient memory` while grids were built. The machine had 15.71 GB
physical RAM and 3.26 GB free at diagnosis. Resuming with two simultaneous
processes completed all failed jobs without changing the inputs. A later
interrupted 3K3J batch also resumed from its checkpoint successfully.

The validated local limit for this workflow is therefore two Vina workers with
two CPU threads each. Larger runs should use a higher-memory remote instance or
an equivalently audited faster docking implementation.

## Score Matrix

| Ligand | Label | 1A9U | 2QD9 | 3K3J | 3KQ7 |
|---|---|---:|---:|---:|---:|
| active L000215 | active | -8.955 | -9.043 | -9.789 | -9.834 |
| active L000255 | active | -7.777 | -6.712 | -7.349 | -6.644 |
| active L000297 | active | -11.030 | -9.822 | -12.300 | -10.120 |
| active L000318 | active | -9.473 | -9.283 | -9.602 | -10.380 |
| decoy L000178 | decoy | -8.935 | -7.471 | -9.385 | -8.591 |
| decoy L003607 | decoy | -8.438 | -8.636 | -9.128 | -9.351 |
| decoy L017349 | decoy | -10.380 | -10.120 | -10.600 | -11.110 |
| decoy L035465 | decoy | -5.526 | -6.097 | -5.672 | -6.538 |

The strong scores for decoy L017349 across all four receptors are a useful
false-positive stress case. DUD-E decoys are presumed inactive rather than
experimentally proven inactive, and Vina score is not binding free energy.

## Decision

The Stage 5 execution gate passes: receptor inputs, ligand preparation,
parallel checkpointing, resume behavior, pose parsing, and matrix construction
work end to end. The next scientifically informative step is a larger
train/validation development benchmark with a search-strength protocol chosen
before looking at validation metrics. No AUC, EF, BEDROC, receptor-selection,
or QUBO conclusion is drawn from this pilot.
