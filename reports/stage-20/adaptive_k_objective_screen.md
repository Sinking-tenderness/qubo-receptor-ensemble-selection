# Stage 20 adaptive receptor-pool cardinality screen

Post-hoc train-only review on MK14 and PPARG. Every k uses the same external robust BEDROC20 comparison metric; no new docking or quantum hardware was used.

## Candidate curve

| k | Mean holdout robust composite | Standard error |
|---:|---:|---:|
| 1 | 0.880836 | 0.014015 |
| 2 | 0.869421 | 0.029724 |
| 3 | 0.850279 | 0.028915 |
| 4 | 0.864589 | 0.033333 |
| 5 | 0.875288 | 0.023995 |
| 6 | 0.863704 | 0.024121 |

- Best observed k: `1`
- One-standard-error smallest k: `1`
- Consecutive-failure stop recommendation: `3`

## Increment rule

| Target | k | Mean increment | Positive folds | Continue? |
|---|---:|---:|---:|---:|
| MK14 | 2 | 0.021339 | 3 | False |
| MK14 | 3 | -0.025880 | 1 | False |
| MK14 | 4 | 0.018696 | 3 | False |
| MK14 | 5 | -0.001081 | 3 | False |
| MK14 | 6 | -0.024272 | 1 | False |
| PPARG | 2 | -0.044169 | 2 | False |
| PPARG | 3 | -0.012403 | 0 | False |
| PPARG | 4 | 0.009923 | 3 | False |
| PPARG | 5 | 0.022480 | 2 | False |
| PPARG | 6 | 0.001102 | 3 | False |

The recommendation is exploratory and cannot amend a prior validation gate or establish quantum advantage.
