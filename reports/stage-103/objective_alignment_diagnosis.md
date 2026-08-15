# Stage103 objective-alignment diagnosis

This is a post-hoc mechanism diagnosis of the frozen Stage99 objective. It does not tune or nominate a replacement objective, initiate docking, unlock PARP1, or authorize quantum hardware.

## k=2 alignment summary

| Target | QUBO vs train BEDROC Spearman | QUBO vs outer BEDROC Spearman | Outer regret of QUBO subset | Oracle k=2 minus k=1 |
| --- | ---: | ---: | ---: | ---: |
| BACE1 | +0.357 | +0.041 | +0.2008 | +0.0288 |
| EGFR | +0.874 | +0.503 | +0.0751 | -0.0282 |
| FA10 | +0.804 | +0.472 | +0.0228 | +0.0091 |
| MK14 | -0.697 | -0.593 | +0.1309 | +0.0857 |
| PPARA | +0.648 | -0.059 | +0.1791 | -0.0135 |
| PPARD | +0.707 | +0.204 | +0.1959 | +0.0070 |
| PPARG | +0.532 | +0.190 | +0.0710 | +0.0039 |

## Decision

If train alignment is high but outer alignment is low, the dominant limitation is transfer/generalization of pair complementarity; do not tune a new objective on these same targets. If train alignment is also low, formulate one separately reviewed surrogate-alignment objective before any untouched-target study.
