# Stage38 stable triplet-residual objective

The objective retains only pair and triplet residuals that recur across scaffold blocks. It is a cubic HUBO that can be reduced exactly to a QUBO with Rosenberg auxiliaries.

| Target | k | Mean fidelity | Mean holdout vs direct classical | Solver-gap cells |
|---|---:|---:|---:|---:|
| MK14 | 3 | 0.9074 | -0.015443 | 0/4 |
| MK14 | 4 | 0.8620 | -0.005487 | 0/4 |
| MK14 | 5 | 0.8224 | +0.011335 | 0/4 |
| MK14 | 6 | 0.7883 | +0.030688 | 0/4 |
| PPARG | 3 | 0.9186 | +0.050981 | 0/4 |
| PPARG | 4 | 0.8717 | +0.013442 | 0/4 |
| PPARG | 5 | 0.8284 | -0.020739 | 0/4 |
| PPARG | 6 | 0.7908 | +0.010461 | 0/4 |

Support gate: **NO-GO**.

Stage38 is a post-hoc objective-development study using only frozen MK14 and PPARG training rows. Its scaffold holdouts are internal development evidence, not independent protein validation. A pass would authorize exact HUBO-to-QUBO quadratization and coefficient-range auditing only. It cannot authorize quantum hardware, claim quantum advantage, start new docking, change prior confirmation failures, or justify further tuning of these frozen weights after outcome inspection.
