# Stage 21: structure-aware conformation-pool QUBO

This is a post-hoc, label-independent structural concept validation.
No docking score, ligand label, fresh-validation row, test row, or quantum hardware result was read.

## Frozen model

`Q(x) = A*(sum(x)-k)^2 - lambda_distance*sum(d_ij*x_i*x_j) + lambda_quality*sum(q_i*x_i)`

The run used lambda_distance=1.0, lambda_quality=0.0, and 24 deterministic restarts per target and k.
The quality term was frozen to zero to isolate structural diversity; invalid structures were removed by a hard preparation gate.

## Selection results

| Target | k | QUBO subset | Mean pair distance | Minimum pair distance | Different from max-min | Different from max-sum | Best restart fraction | Unique restart solutions |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MK14 | 2 | `MK14_2BAJ_aligned+MK14_3ZSG_aligned` | 1.000000 | 1.000000 | true | false | 1.000 | 1 |
| MK14 | 3 | `MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3ZSG_aligned` | 0.913344 | 0.783759 | false | false | 0.833 | 2 |
| MK14 | 4 | `MK14_1OZ1_aligned+MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3ZSG_aligned` | 0.867828 | 0.692037 | true | false | 1.000 | 1 |
| MK14 | 6 | `MK14_1OZ1_aligned+MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3MPT_aligned+MK14_3ZSG_aligned+MK14_4FA2_aligned` | 0.823311 | 0.615398 | true | true | 0.750 | 2 |
| MK14 | 8 | `MK14_1A9U_aligned+MK14_1OZ1_aligned+MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3MPT_aligned+MK14_3ZSG_aligned+MK14_3ZSI_aligned+MK14_4FA2_aligned` | 0.800156 | 0.524685 | true | false | 0.875 | 2 |
| PPARG | 2 | `PPARG_6AD9_aligned+PPARG_8B8X_aligned` | 1.000000 | 1.000000 | true | true | 1.000 | 1 |
| PPARG | 3 | `PPARG_5F9B_aligned+PPARG_6AD9_aligned+PPARG_8B8X_aligned` | 0.982543 | 0.969120 | true | false | 1.000 | 1 |
| PPARG | 4 | `PPARG_3CS8_aligned+PPARG_5F9B_aligned+PPARG_6AD9_aligned+PPARG_8B8X_aligned` | 0.935229 | 0.762073 | true | false | 1.000 | 1 |
| PPARG | 6 | `PPARG_2HFP_aligned+PPARG_2Q6S_aligned+PPARG_3CS8_aligned+PPARG_5F9B_aligned+PPARG_6AD9_aligned+PPARG_8B8X_aligned` | 0.852093 | 0.663220 | true | true | 1.000 | 1 |
| PPARG | 8 | `PPARG_2HFP_aligned+PPARG_2Q6S_aligned+PPARG_3CS8_aligned+PPARG_5F9B_aligned+PPARG_5UGM_aligned+PPARG_6AD9_aligned+PPARG_8B8X_aligned+PPARG_8FHF_aligned` | 0.798186 | 0.553633 | true | false | 1.000 | 1 |

## Interpretation boundary

A different or more stable structural subset is evidence that the selection problem can be moved into a combinatorial objective. It is not evidence of better virtual-screening enrichment, quantum advantage, or biological superiority.
The next gate is to redock only a preregistered, small matched subset if and only if the structural QUBO produces a reproducible difference from max-min and max-sum baselines.
