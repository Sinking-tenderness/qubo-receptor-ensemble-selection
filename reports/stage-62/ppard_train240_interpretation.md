# Stage62 PPARD Train-240 interpretation

## Result

The frozen transferred rank-pair QUBO did not pass its preregistered PPARD
application-support gate.

| Nested outer-fold method | Mean robust BEDROC20 |
|---|---:|
| Linear Top-k | 0.807990 |
| Best single receptor | 0.800143 |
| Exact transferred QUBO | 0.777450 |
| Direct BEDROC greedy | 0.772460 |

The exact QUBO lost 0.022692 to the best-single comparator and 0.030540 to
linear Top-k, while exceeding direct BEDROC greedy by 0.004990. Only one of
four outer folds improved over the best single receptor. The final frozen
one-standard-error rule selected k=1.

## Mechanistic reading

Three outer folds selected k=1. Outer fold 2 selected k=5 and then generalized
poorly, which explains most of the negative transfer. Exact QUBO and the
beam-64 multi-start swap search selected the same optimum in all 102 fitted
objective cells. Direct QUBO greedy missed the exact objective in 32 cells.

The quadratic landscape is therefore nontrivial for direct greedy, but not yet
hard for the frozen strong classical baseline at 29 receptors and k<=6.

## Boundary and next action

No fresh-validation or locked-test row was read, no docking was added, and no
QUBO coefficient was changed after Stage60. PPARD outcome fitting must now
stop. The result is a prospective negative transfer test, not evidence that
all QUBO formulations or quantum implementations are ineffective.

The next defensible research step is a cross-target, development-only failure
analysis that asks why the pair term overestimates multi-receptor benefit in
the unstable fold. Any revised objective must be designed without further
PPARD fitting and preregistered on a new target before claiming efficacy.
