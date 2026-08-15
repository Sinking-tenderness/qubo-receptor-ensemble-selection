# Stage 23: QUBO sampler stability

Post-hoc structure-only diagnostic using independent fixed-cardinality simulated-annealing reads.

Frozen objective: `coverage + 0.15 * mean structural diversity`; `k=8`; primary neighborhood fraction `0.10`.

| Target | Fraction | Best batch objective | Strong classical | Within tolerance | Above strong classical |
|---|---:|---:|---:|---:|---:|
| MK14 | 0.05 | 0.976243 | 0.976243 | 1.00 | 0.00 |
| MK14 | 0.10 | 1.107438 | 1.104867 | 1.00 | 1.00 |
| MK14 | 0.20 | 1.113705 | 1.113705 | 1.00 | 0.00 |
| PPARG | 0.05 | 0.872471 | 0.872471 | 1.00 | 0.00 |
| PPARG | 0.10 | 1.071642 | 1.066192 | 1.00 | 1.00 |
| PPARG | 0.20 | 1.104990 | 1.104817 | 1.00 | 1.00 |

Sampler stability gate passed: `false`.
A pass authorizes only a separate small matched Uni-Dock preregistration; it does not establish quantum advantage or authorize hardware.

This stage can establish only objective-level stability of a classical simulated-annealing sampler for the frozen structural QUBO and relative performance against the recorded classical baselines. It cannot establish quantum speedup, quantum advantage, docking enrichment, prospective generalization, or biological benefit. A passing gate authorizes only a separately preregistered small matched Uni-Dock experiment.
