# Stage62 PPARD Train-240 frozen nested QUBO analysis

## Nested outer-fold performance

| Method | Mean robust BEDROC20 |
|---|---:|
| Exact transferred QUBO | 0.777450 |
| Best single receptor | 0.800143 |
| Linear Top-k | 0.807990 |
| Direct BEDROC greedy | 0.772460 |

## Frozen decisions

- Final one-standard-error k: 1.
- Application support gate: NO-GO.
- Solver novelty gate: NO-GO.
- Fresh validation authorized: False.
- Quantum hardware and quantum-advantage claims remain unauthorized.

Stage62 is the prospectively frozen PPARD development application test authorized by Stage59 and specified by Stage60. It may decide whether one untouched PPARD fresh-validation run is warranted. It cannot access fresh validation or locked test rows, retune the QUBO, establish independent efficacy, authorize quantum hardware, or claim quantum speedup or quantum advantage.
