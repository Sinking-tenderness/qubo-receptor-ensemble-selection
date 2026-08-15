# Stage100 adaptive stopping QUBO

The receptor count is no longer fixed. The primary rule selects the smallest k within one standard error of the best inner-fold BEDROC.

| Target | Adaptive BEDROC | Single BEDROC | Gain |
|---|---:|---:|---:|
| MK14 | 0.383556 | 0.370233 | +0.013323 |
| PPARG | 0.878428 | 0.887457 | -0.009029 |
| BACE1 | 0.962519 | 0.983155 | -0.020635 |
| PPARA | 0.815153 | 0.832194 | -0.017041 |
| PPARD | 0.697470 | 0.697470 | +0.000000 |

Mean gain: `-0.006676`

Worst-target gain: `-0.020635`

Nontrivial selections: `7/25`

Go/No-Go: `NO-GO`
