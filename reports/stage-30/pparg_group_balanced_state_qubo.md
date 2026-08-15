# Stage 30: PPARG group-balanced multiscale-state QUBO

Frozen objective: 0.3 within-start centrality + 0.5 cross-start distance + 0.2 multiscale state separation; exactly one frame per MD start.

| Pool | n | Search states | Strong classical | Annealing | Delta | Stable | Exact gap | Variables/couplers |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| m=2 | 16 | 256 | 0.70499286 | 0.70499286 | 0 | True | 0 | 16/120 |
| m=4 | 32 | 65536 | 0.71054901 | 0.71054901 | 0 | True | 0 | 32/496 |
| m=8 | 64 | 16777216 | 0.71252121 | 0.71252121 | 0 | True | NA | 64/2016 |
| m=16 | 128 | 4294967296 | 0.71982325 | 0.71982325 | 0 | True | NA | 128/8128 |
| m=32 | 256 | 1099511627776 | 0.73059971 | 0.73059971 | 0 | True | NA | 256/32640 |
| m=64 | 512 | 281474976710656 | 0.73840555 | 0.73840555 | 0 | True | NA | 512/130816 |
| m=100 | 800 | 10000000000000000 | 0.74136769 | 0.74136769 | 0 | True | NA | 800/319600 |
| m=150 | 1200 | 256289062500000000 | 0.74471779 | 0.74471779 | 0 | True | NA | 1200/719400 |

Construction/equivalence/exactness gates: **PASS**.
Annealing stability gate: **PASS**.
Solver novelty gate: **NO-GO** (0 strict wins).
Direct-QPU readiness gate: **NO-GO**.

No docking scores, ligand labels, validation/test rows, new docking jobs, or quantum-hardware outputs were used.
