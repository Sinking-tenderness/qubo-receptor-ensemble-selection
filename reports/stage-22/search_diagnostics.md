# Stage 22 search diagnostics

This report is post-hoc and structure-only. It does not revise the frozen Stage 22 gate and does not authorize docking or quantum-hardware execution.

## Frozen Stage 22 result

The structural-state coverage objective improved over direct greedy selection at the primary `10% / k=8` setting on both targets, but the exact-subset restart agreement gate failed.

| Target | Stage22 candidate | Direct greedy | Candidate delta | Exact-subset restart frequency |
|---|---:|---:|---:|---:|
| MK14 | 1.107438 | 1.100730 | +0.006708 | 1/32 |
| PPARG | 1.071642 | 1.055606 | +0.016037 | 2/32 |

## Stronger classical search

Deterministic beam search retained up to 2,048 partial subsets. Every final beam result was then refined to a one-swap local optimum using the same frozen objective.

| Target | Neighborhood | Stage22 candidate | Best beam plus swap | Candidate minus strongest beam |
|---|---:|---:|---:|---:|
| MK14 | 5% | 0.976243 | 0.976243 | 0.000000 |
| MK14 | 10% | 1.107438 | 1.104867 | +0.002571 |
| MK14 | 20% | 1.113705 | 1.113705 | 0.000000 |
| PPARG | 5% | 0.872471 | 0.872471 | 0.000000 |
| PPARG | 10% | 1.071642 | 1.066192 | +0.005450 |
| PPARG | 20% | 1.104990 | 1.104817 | +0.000174 |

The primary setting retains a positive candidate advantage on both targets after this stronger baseline. The sensitivity settings are mostly tied or nearly tied, so the result is not robust enough for a QUBO-superiority claim.

## Global MILP attempt

The frozen objective was also linearized as a binary MILP with structural-state coverage variables and pair-product variables. Under a 15-second controlled limit, all six jobs reached the time limit with relative gaps from `0.162` to `14.948`; all incumbents were inferior to known greedy or multistart solutions. These runs establish only that a generic MILP oracle is not a practical global verifier at this scale. They do not estimate the global optimum.

## Decision

The frozen Stage 22 gate remains failed. No matched docking or quantum-hardware job is authorized.

The result nevertheless identifies a narrower, testable next step: stabilize optimization of the frozen QUBO objective with independent sampler batches and require objective-level convergence plus superiority over direct greedy and beam-plus-swap on both targets. Only a prospective pass should authorize a small matched Uni-Dock experiment.
