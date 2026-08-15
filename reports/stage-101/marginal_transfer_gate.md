# Stage101 marginal transfer gate

Stage101 asks whether inner-fold evidence predicts the held-out value of adding another receptor. It does not tune a new QUBO on held-out labels.

| Transition | Spearman | p-value | Sign accuracy | False positives |
|---|---:|---:|---:|---:|
| k=1 to 2 | -0.472 | 0.0171 | 36.00% | 8 |
| k=2 to 3 | +0.359 | 0.0778 | 52.00% | 7 |
| All | -0.188 | 0.1920 | 44.00% | 15 |

## Policy outcomes

| Policy | Mean target gain | Worst target gain | Nontrivial folds |
|---|---:|---:|---:|
| lcb_z0p0 | -0.013687 | -0.070444 | 12/25 |
| lcb_z0p5 | -0.013641 | -0.070444 | 11/25 |
| lcb_z1p0 | -0.011982 | -0.053403 | 8/25 |
| lcb_z1p64 | -0.005933 | -0.020635 | 3/25 |
| lcb_z1p96 | -0.004127 | -0.020635 | 1/25 |
| all_inner_positive | -0.009599 | -0.020635 | 4/25 |
| two_of_three_positive | -0.003368 | -0.062203 | 15/25 |
| loto_ridge | -0.005885 | -0.101520 | 16/25 |
| always_single | +0.000000 | +0.000000 | 0/25 |
| outer_oracle_k | +0.041197 | +0.000000 | 13/25 |

LOTO candidate gate: `NO-GO`

Outer-oracle adaptive-k ceiling: `+0.041197` mean target gain.

Interpretation: a useful variable-k solution exists in these folds, but the current inner-fold marginal signal does not identify it reliably. No more threshold tuning is allowed on these same matrices.
