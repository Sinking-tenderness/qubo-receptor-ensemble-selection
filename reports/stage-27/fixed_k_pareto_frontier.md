# Stage 27: fixed-k Pareto frontier

Frozen benefit: 0.25 single coverage + 0.75 double coverage + 0.10 pair diversity with one common k=8 pair denominator. No per-conformer cost is optimized.

| Target | n | Supported k under nonnegative cost | Strict sampler wins | Stable cells | Direct-QPU-ready cells |
|---|---:|---|---:|---:|---:|
| MK14 | 43 | 0,7,8 | 0 | 8/8 | 0/8 |
| PPARG | 98 | 0,2,3,6,7,8 | 0 | 8/8 | 0/8 |
| BACE1 | 49 | 0,2,4,6,8 | 0 | 8/8 | 0/8 |
| EGFR | 18 | 0,1,2,4,6,8 | 0 | 8/8 | 8/8 |
| FA10 | 19 | 0,8 | 0 | 8/8 | 8/8 |

Frontier-validity gate: **PASS**.
Solver-novelty gate: **NO-GO**.
Direct-QPU readiness gate: **NO-GO**.

No docking scores, ligand labels, fresh-validation rows, test rows, or quantum-hardware outputs were read.
