# Stage 26: variable-budget two-hit consensus QUBO

Frozen structure-only objective: 0.25 single coverage + 0.75 double coverage + 0.10 pair diversity - 0.04 per selected conformer; at most eight conformers.

| Target | n | Full | Selected k | Sampler | Strong classical | Delta | Stable | QUBO variables | Direct QPU ready |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| MK14 | 16 | no | 8 | 0.316944 | 0.316944 | 0.000000 | 1.00 | 100 | yes |
| MK14 | 24 | no | 8 | 0.442015 | 0.442015 | 0.000000 | 1.00 | 172 | yes |
| MK14 | 32 | no | 8 | 0.506693 | 0.506693 | 0.000000 | 1.00 | 260 | no |
| MK14 | 43 | yes | 8 | 0.483644 | 0.483644 | 0.000000 | 1.00 | 391 | no |
| PPARG | 16 | no | 8 | 0.274973 | 0.274973 | 0.000000 | 1.00 | 100 | yes |
| PPARG | 24 | no | 8 | 0.551511 | 0.551511 | 0.000000 | 1.00 | 172 | yes |
| PPARG | 32 | no | 8 | 0.615487 | 0.615487 | 0.000000 | 1.00 | 260 | no |
| PPARG | 64 | no | 8 | 0.564409 | 0.564409 | 0.000000 | 1.00 | 580 | no |
| PPARG | 98 | yes | 8 | 0.562542 | 0.562542 | 0.000000 | 1.00 | 1082 | no |
| BACE1 | 16 | no | 8 | 0.355052 | 0.355052 | 0.000000 | 1.00 | 100 | yes |
| BACE1 | 24 | no | 8 | 0.439446 | 0.439446 | 0.000000 | 1.00 | 172 | yes |
| BACE1 | 32 | no | 8 | 0.474913 | 0.474913 | 0.000000 | 1.00 | 260 | no |
| BACE1 | 49 | yes | 8 | 0.441304 | 0.441304 | 0.000000 | 1.00 | 445 | no |

Optimization-novelty gate: **NO-GO**.
Direct-QPU readiness under frozen thresholds: **NO-GO**.

No docking scores, ligand labels, fresh validation rows, test rows, or quantum-hardware outputs were read.
