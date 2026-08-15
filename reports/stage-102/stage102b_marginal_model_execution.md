# Stage102B marginal-model execution

Stage102B operationalizes the two candidate rules named in Stage102. It is a posthoc development amendment, not a PARP1 confirmation or hardware release.

## Candidate outcomes

| Policy | Mean target gain | Worst target gain | Positive targets at +0.02 | Nontrivial folds | New-target sign accuracy | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mechanistic_bootstrap_lcb | -0.006691 | -0.053403 | 0 | 3 | 65.00% | NO-GO |
| target_held_out_l2_ridge | -0.015523 | -0.130063 | 1 | 15 | 55.00% | NO-GO |

## Exact-QUBO target outcomes

| Target | Policy | BEDROC20 | Gain over single | Selected k |
| --- | --- | ---: | ---: | --- |
| BACE1 | mechanistic_bootstrap_lcb | 0.983155 | +0.000000 | 1|1|1|1|1 |
| BACE1 | target_held_out_l2_ridge | 0.974690 | -0.008465 | 1|1|2|2|2 |
| EGFR | mechanistic_bootstrap_lcb | 0.371583 | +0.000000 | 1|1|1|1|1 |
| EGFR | target_held_out_l2_ridge | 0.358613 | -0.012969 | 1|1|1|2|2 |
| FA10 | mechanistic_bootstrap_lcb | 0.733190 | +0.006566 | 1|1|1|1|2 |
| FA10 | target_held_out_l2_ridge | 0.734259 | +0.007636 | 2|3|1|1|1 |
| MK14 | mechanistic_bootstrap_lcb | 0.370233 | +0.000000 | 1|1|1|1|1 |
| MK14 | target_held_out_l2_ridge | 0.370233 | +0.000000 | 1|1|1|1|1 |
| PPARA | mechanistic_bootstrap_lcb | 0.778791 | -0.053403 | 1|2|2|1|1 |
| PPARA | target_held_out_l2_ridge | 0.702132 | -0.130063 | 3|2|3|3|3 |
| PPARD | mechanistic_bootstrap_lcb | 0.697470 | +0.000000 | 1|1|1|1|1 |
| PPARD | target_held_out_l2_ridge | 0.730061 | +0.032591 | 2|1|1|1|2 |
| PPARG | mechanistic_bootstrap_lcb | 0.887457 | +0.000000 | 1|1|1|1|1 |
| PPARG | target_held_out_l2_ridge | 0.890067 | +0.002609 | 2|1|1|1|1 |

## Decision

Reject this operationalization of adaptive-cardinality development. Do not prepare or dock PARP1 and do not run quantum hardware.

No new docking, PARP1 rows, locked-test rows, or quantum-hardware jobs were used.
