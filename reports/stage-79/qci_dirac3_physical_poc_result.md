# Stage79 QCI Dirac-3 Physical PoC Result

## Primary result

The frozen physical-hardware endpoint passed. Dirac-3 returned the certified
float64 optimum for all `500/500` confirmation samples. The two PPARG positive
controls improved their classical warm states in `100/100` samples each. The
BACE1, PPARA, and PPARD negative controls produced `0/300` false improvements.
All `500/500` returned samples were feasible, and no sample fell below an
independently certified optimum.

| Instance | Role | Optimum hits | Improvement hits | Warm to optimum delta |
|---|---|---:|---:|---:|
| PPARG OF3, k=16 | positive | 100/100 | 100/100 | -0.242954 |
| PPARG OF1, k=16 | positive | 100/100 | 100/100 | -0.054912 |
| BACE1 OF0, k=16 | negative | 100/100 | 0/100 | 0.000000 |
| PPARA OF2, k=8 | negative | 100/100 | 0/100 | 0.000000 |
| PPARD OF0, k=12 | negative | 100/100 | 0/100 | 0.000000 |

## Calibration and resource use

All four calibration schedules recovered the diagnostic optimum in `25/25`
samples. Schedule 1 was selected by the frozen tie-break because it used 15
device seconds, versus 31, 29, and 62 seconds for schedules 2, 3, and 4.
Calibration and confirmation used 434 recorded device seconds in total, leaving
166 seconds in the unpaid trial allocation.

## Independent audit

The five raw response vectors were decoded again against the original Stage78
float64 BQMs and move tables. Device-reported energies were not used to classify
the result. Both positive optima contain one receptor swap; all three negative
optima are the all-zero no-swap state. Raw counts, feasibility, exact energies,
and the positive/negative endpoint labels all matched the frozen certificates.
No API-token marker was observed in the result archive or log.

## Scientific interpretation

This is a successful cross-hardware physical optimization proof of concept for
protein-derived QUBOs. It demonstrates that the frozen local receptor-selection
landscape can be translated to Dirac-3 without changing its optimum and that the
hardware distinguishes known improvable and non-improvable controls.

It is not evidence of quantum advantage. The confirmation problems contain only
38-40 binary variables, and the observed optima are either a single swap or no
swap. Exact MILP and ordinary local search solve these instances readily. The
result also does not establish biological generalization, discover a new drug,
or demonstrate end-to-end virtual-screening acceleration.

The next defensible experiment is a frozen scaling study with the same
protein-derived construction: increase the local move neighborhood, compare
Dirac-3 against exact MILP, simulated annealing, tabu/local search, and random
sampling under matched solution-quality and time-to-solution definitions, and
reserve an untouched set of positive and negative controls.
