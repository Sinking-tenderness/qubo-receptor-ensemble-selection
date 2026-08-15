# Stage 19h auxiliary-variable coverage QUBO

## Scope

Post-hoc nested scaffold-CV development on MK14 and PPARG train matrices only. No new docking or protected-panel row was read.

## Results

| Target | Method | Mean holdout robust composite | Mean primary |
|---|---|---:|---:|
| MK14 | additive_nested | 0.933670 | 0.933489 |
| MK14 | auxiliary_coverage_nested | 0.920731 | 0.922242 |
| MK14 | direct_greedy | 0.919450 | 0.922534 |
| MK14 | stable_singleton_linear | 0.918692 | 0.921664 |
| MK14 | v1_qubo_exact | 0.937928 | 0.940242 |
| PPARG | additive_nested | 0.902152 | 0.912390 |
| PPARG | auxiliary_coverage_nested | 0.861018 | 0.871704 |
| PPARG | direct_greedy | 0.848095 | 0.850126 |
| PPARG | stable_singleton_linear | 0.890256 | 0.905514 |
| PPARG | v1_qubo_exact | 0.807593 | 0.794441 |

## Decision

- Development gate passed: `False`
- BACE1 method amendment authorized: `False`
- Next gate: `do_not_spend_new_docking_budget_on_this_coverage_objective; review_quantum_application_scope`

## Encoding

- `x_i` selects a receptor.
- `y_a` marks an active ligand covered by at least one selected receptor.
- `z_d` marks a decoy exposed by at least one selected receptor.
- Binary slack variables enforce the active-ligand OR relation as a quadratic equality penalty.
- Decoy exposure constraints use quadratic implication penalties.

Stage 19h is post-hoc train-only development over MK14 and PPARG. It does not create independent validation, does not modify prior failed gates, does not read BACE1, fresh-validation, or locked-test rows, and does not demonstrate quantum execution or quantum advantage. Its purpose is to decide whether a constraint-faithful auxiliary QUBO is worth a separately preregistered external test.
