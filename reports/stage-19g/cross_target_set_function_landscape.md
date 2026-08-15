# Stage 19g cross-target set-function landscape

## Scope

Post-hoc diagnostic on MK14 and PPARG training matrices only. No new docking, BACE1 docking, fresh-validation, or test row was read.

## Target decision

| Target | k=3 gap folds | Qualifying sizes | Pair top-1% overlap | Pair regret | Residual stability | Route |
|---|---:|---|---:|---:|---:|---|
| MK14 | 0/4 | none | 0.417 | 0.001293 | 0.755 | no_stable_greedy_gap_for_efficacy_claim |
| PPARG | 0/4 | none | 0.208 | 0.005418 | 0.837 | no_stable_greedy_gap_for_efficacy_claim |

## Set-function structure

| Target | Submodularity violations | Negative marginals | Legacy k=3 train gap | Legacy k=3 holdout delta |
|---|---:|---:|---:|---:|
| MK14 | 0.390 | 0.312 | 0.000302 | +0.017883 |
| PPARG | 0.503 | 0.469 | 0.004824 | -0.006236 |

## Greedy gaps by size

| Target | k | Mean train gap | Gap folds | Mean holdout delta |
|---|---:|---:|---:|---:|
| MK14 | 1 | 0.000000 | 0/4 | +0.000000 |
| MK14 | 2 | 0.000418 | 0/4 | +0.017761 |
| MK14 | 3 | 0.000302 | 0/4 | +0.017883 |
| MK14 | 4 | 0.000132 | 0/4 | +0.011981 |
| MK14 | 5 | 0.000362 | 0/4 | +0.012615 |
| MK14 | 6 | 0.001733 | 1/4 | +0.025231 |
| PPARG | 1 | 0.000000 | 0/4 | +0.000000 |
| PPARG | 2 | 0.000000 | 0/4 | +0.000000 |
| PPARG | 3 | 0.000000 | 0/4 | +0.000000 |
| PPARG | 4 | 0.001038 | 1/4 | +0.017794 |
| PPARG | 5 | 0.001403 | 1/4 | +0.015955 |
| PPARG | 6 | 0.000549 | 1/4 | +0.018392 |

## Decision

- Cross-target route: `no_cross_target_efficacy_qubo_route_authorized`
- BACE1 method amendment authorized: `False`
- Next stage: `review_target_selection_or_quantum_application_only_claim`

## Boundary

Stage 19g is a post-hoc mathematical diagnostic over MK14 and PPARG training matrices whose outcomes are already known. It adds no docking and reads no BACE1 docking, fresh-validation, or test row. It may choose the representation class for one subsequent train-only development stage, but it cannot amend BACE1, alter prior failures, establish cross-target QUBO superiority, demonstrate quantum execution, or establish quantum advantage.
