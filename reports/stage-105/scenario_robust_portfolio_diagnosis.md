# Stage105 scenario-robust portfolio diagnosis

For every outer training split, Stage105 requires the selected subset to match or exceed the full-train pair-off baseline's singleton utility in each of four frozen scaffold-jackknife scenarios. It then minimizes Stage68 three-seed stable redundancy. This is posthoc mechanism diagnosis only.

| Target | k | Changed folds | Robust BEDROC gain | Redundancy reduction | Worst gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| BACE1 | 2 | 0/4 | +0.000000 | +0.000000 | +0.000000 |
| BACE1 | 3 | 0/4 | +0.000000 | +0.000000 | +0.000000 |
| EGFR | 2 | 0/5 | +0.000000 | +0.000000 | +0.000000 |
| EGFR | 3 | 0/5 | +0.000000 | +0.000000 | +0.000000 |
| FA10 | 2 | 0/5 | +0.000000 | +0.000000 | +0.000000 |
| FA10 | 3 | 0/5 | +0.000000 | +0.000000 | +0.000000 |
| PPARA | 2 | 0/4 | +0.000000 | +0.000000 | +0.000000 |
| PPARA | 3 | 0/4 | +0.000000 | +0.000000 | +0.000000 |
| PPARD | 2 | 0/4 | +0.000000 | +0.000000 | +0.000000 |
| PPARD | 3 | 0/4 | +0.000000 | +0.000000 | +0.000000 |
| PPARG | 2 | 0/4 | +0.000000 | +0.000000 | +0.000000 |
| PPARG | 3 | 0/4 | +0.000000 | +0.000000 | +0.000000 |

## Decision

Do not retune this scenario constraint on the same six targets. Use this result only to decide whether a separately reviewed untouched-target protocol is scientifically justified; no current protected dataset or hardware task is released.
