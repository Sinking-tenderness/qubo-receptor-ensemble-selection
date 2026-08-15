# Stage 29: PPARG MD pure-QUBO solver scaling

Frozen structure-only objective: 0.4 mean centrality + 0.6 mean pair distance - 0.1 mean same-trajectory temporal redundancy; k=8.

| Pool | Class | n | Strong classical | Annealing | Delta | Stable | Exact gap | QUBO variables/couplers |
|---|---|---:|---:|---:|---:|---|---:|---:|
| primary_n0016 | primary_scaling | 16 | 0.55631147 | 0.55631147 | 0 | True | 0 | 16/120 |
| primary_n0032 | primary_scaling | 32 | 0.56683886 | 0.56683886 | 0 | True | NA | 32/496 |
| primary_n0064 | primary_scaling | 64 | 0.57224349 | 0.57224349 | 0 | True | NA | 64/2016 |
| primary_n0120 | primary_scaling | 120 | 0.57786082 | 0.57786082 | 0 | True | NA | 120/7140 |
| primary_n0240 | primary_scaling | 240 | 0.58510341 | 0.58510341 | 0 | False | NA | 240/28680 |
| primary_n0480 | primary_scaling | 480 | 0.59255578 | 0.59232201 | -0.00023376678 | False | NA | 480/114960 |
| primary_n0800 | primary_scaling | 800 | 0.59394189 | 0.59265845 | -0.0012834432 | False | NA | 800/319600 |
| primary_n1200 | primary_scaling | 1200 | 0.59939602 | 0.59866607 | -0.00072995587 | False | NA | 1200/719400 |
| uniform_100ps_n240 | sensitivity | 240 | 0.58546559 | 0.58546559 | 0 | False | NA | 240/28680 |
| uniform_200ps_n120 | sensitivity | 120 | 0.57587538 | 0.57587538 | 0 | True | NA | 120/7140 |
| exclude_3d6d_n1050 | sensitivity | 1050 | 0.58820238 | 0.58599190 | -0.0022104745 | False | NA | 1050/550725 |

Scaling/equivalence/exactness gates: **PASS**.
Annealing stability gate: **NO-GO**.
Solver novelty gate: **NO-GO** (0 primary strict wins).
Direct-QPU readiness gate: **NO-GO**.

The 3D6D exclusion is post-hoc sensitivity evidence and does not affect the primary gate. No docking scores, ligand labels, validation/test rows, new docking jobs, or quantum hardware outputs were used.
