# Stage 19f scaffold-stable pair QUBO

## Scope

This is nested scaffold-CV development evidence on MK14 and PPARG train rows only.
Pair terms survive only when their cross-block lower confidence bound is positive.
No BACE1 docking, fresh-validation, or locked-test row was read.

## Results

| Target | Method | Mean composite | Worst fold | Mean primary |
|---|---|---:|---:|---:|
| MK14 | additive_nested | 0.933670 | 0.895148 | 0.933489 |
| MK14 | direct_greedy | 0.919450 | 0.883354 | 0.922534 |
| MK14 | stable_pair_nested | 0.903561 | 0.819800 | 0.908244 |
| MK14 | stable_singleton_linear | 0.918692 | 0.878888 | 0.921664 |
| MK14 | v1_qubo_exact | 0.937928 | 0.895148 | 0.940242 |
| PPARG | additive_nested | 0.902152 | 0.824426 | 0.912390 |
| PPARG | direct_greedy | 0.848095 | 0.635079 | 0.850126 |
| PPARG | stable_pair_nested | 0.831541 | 0.707236 | 0.843138 |
| PPARG | stable_singleton_linear | 0.890256 | 0.781905 | 0.905514 |
| PPARG | v1_qubo_exact | 0.807593 | 0.635079 | 0.794441 |

## Comparisons

- stable_pair_nested minus direct_greedy: mean -0.016221, wins 3/8.
- stable_pair_nested minus additive_nested: mean -0.050360, wins 2/8.
- stable_pair_nested minus v1_qubo_exact: mean -0.005209, wins 3/8.
- stable_pair_nested minus stable_singleton_linear: mean -0.036923, wins 1/8.

## Decision

- A retained nonzero pair term participated in the selected triple in 5/8 outer folds.
- The meaningful-pair presence check passed, but every performance comparison check failed.
- Status: `stage19f_stable_pair_qubo_not_supported_do_not_amend_bace1`
- BACE1 amendment authorized: `False`

## Boundary

Stage 19f is post-hoc development evidence built after MK14 and PPARG outcomes were known. Nested scaffold-CV can reject unstable pair terms but cannot create independent validation. This stage reads no BACE1 docking, fresh-validation, or locked-test row; it cannot alter prior failures, establish QUBO superiority, demonstrate quantum execution, or establish quantum advantage.
