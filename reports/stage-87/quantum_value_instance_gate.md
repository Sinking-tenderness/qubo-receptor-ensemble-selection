# Stage87 quantum-value instance gate

## Prior evidence review

- Stage37-68 repeatedly tested whether quadratic or higher-order receptor interactions improve held-out screening. No objective produced a stable cross-target superiority claim.
- Stage73-75 showed that strong classical methods match every available certified exact fixed-k or variable-k reference.
- Stage79 physically executed local protein-derived QUBOs, but Stage80 found no multi-move local trap among 100 canonical subproblems.
- Stage85-86 showed that the scientifically meaningful global constrained model is not faithfully sampled by the current Dirac-3 penalty interface.

## Targeted supplement

Stage68 contained 4 post-hoc cells where the exact portfolio differs from direct greedy and greedy-swap, has a strictly better redundancy objective, and improves primary, mean-seed, worst-seed, and robust holdout BEDROC.

| Target/fold | k | Candidate | Total states | Feasible states | Exact over swap BEDROC | Enumeration seconds |
|---|---:|---|---:|---:|---:|---:|
| PPARA/2 | 6 | uncertainty_1p0x | 38,760 | 85 | +0.034158 | 0.397 |
| PPARA/3 | 6 | uncertainty_0p25x | 38,760 | 23 | +0.116287 | 0.486 |
| PPARA/3 | 6 | uncertainty_1p0x | 38,760 | 415 | +0.055678 | 0.420 |
| PPARD/3 | 3 | uncertainty_1p0x | 3,654 | 57 | +0.030628 | 0.034 |

All four candidates are complete-enumeration problems with at most 38,760 states. They verify that one deterministic greedy start can be trapped, but they do not establish classical difficulty.

## Decision

The strict instance gate fails. Constraint-preserving QAOA simulation and new hardware jobs remain blocked. The next scientific task is to preregister a larger independently validated decision problem where the certified global solution has both biological benefit and a demonstrated gap over strong classical search.
