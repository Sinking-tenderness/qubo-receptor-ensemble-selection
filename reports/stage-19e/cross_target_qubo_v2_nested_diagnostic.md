# Stage 19e cross-target QUBO v2 diagnostic

## Scope

This is nested scaffold-CV development evidence on MK14 and PPARG train rows only.
The inner folds choose ridge alpha; outer folds report performance.
No fresh-validation or locked-test row was read.

## Outer-fold results

| Target | Method | Mean composite | Worst fold | Mean primary | Mean rank |
|---|---|---:|---:|---:|---:|
| MK14 | additive_nested | 0.933670 | 0.895148 | 0.933489 | 28.00 |
| MK14 | composite_exact | 0.937333 | 0.895148 | 0.939831 | 43.75 |
| MK14 | direct_exact | 0.937333 | 0.895148 | 0.939831 | 43.75 |
| MK14 | direct_greedy | 0.919450 | 0.883354 | 0.922534 | 62.00 |
| MK14 | holdout_oracle | 0.964792 | 0.947851 | 0.966662 | 1.00 |
| MK14 | quadratic_nested | 0.937333 | 0.895148 | 0.939831 | 43.75 |
| MK14 | v1_qubo_exact | 0.937928 | 0.895148 | 0.940242 | 38.75 |
| PPARG | additive_nested | 0.902152 | 0.824426 | 0.912390 | 49.00 |
| PPARG | composite_exact | 0.841859 | 0.635079 | 0.841644 | 166.50 |
| PPARG | direct_exact | 0.870062 | 0.747892 | 0.877719 | 106.25 |
| PPARG | direct_greedy | 0.848095 | 0.635079 | 0.850126 | 151.25 |
| PPARG | holdout_oracle | 0.942166 | 0.885462 | 0.954870 | 1.00 |
| PPARG | quadratic_nested | 0.845076 | 0.635079 | 0.849674 | 160.50 |
| PPARG | v1_qubo_exact | 0.807593 | 0.635079 | 0.794441 | 261.75 |

## Paired comparisons

- quadratic_nested minus direct_greedy: mean +0.007432, wins 2/8.
- quadratic_nested minus additive_nested: mean -0.026706, wins 2/8.
- quadratic_nested minus v1_qubo_exact: mean +0.018444, wins 2/8.

## Decision

- Status: `stage19e_quadratic_v2_not_supported_do_not_amend_bace1`
- BACE1 amendment authorized: `False`

## Boundary

Stage 19e is post-hoc development evidence because the MK14 and PPARG outcomes were already known. Nested scaffold-CV reduces alpha-selection bias but does not create independent validation. The analysis may authorize a prospectively frozen secondary BACE1 method only before any BACE1 benchmark docking; it cannot alter prior failures, use BACE1 outcomes, establish QUBO superiority, demonstrate quantum execution, or establish quantum advantage.
