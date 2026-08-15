# Stage 19d PPARG Train-668 comparator analysis

## Scope

This is scaffold-grouped four-fold out-of-fold development evidence.
The PPARG pair-synergy QUBO v1 weights were transferred without retuning.
No fresh-validation or locked-test row was read.

## OOF results

| Method | BEDROC20 | Mean seed | Worst seed | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| edock_rf_all16 | 0.9688 | 0.9660 | 0.9604 | 0.8997 | 0.8949 |
| edock_rf_rfe3 | 0.9245 | 0.9363 | 0.9123 | 0.8670 | 0.8504 |
| ricci_lr_all16 | 0.9485 | 0.9375 | 0.9099 | 0.8899 | 0.8770 |
| matched_linear_top3 | 0.8993 | 0.9101 | 0.9041 | 0.8420 | 0.8238 |
| enopt_xgboost_all16 | 0.9244 | 0.9181 | 0.9033 | 0.8810 | 0.8614 |
| hantz_auc_top3_min | 0.9189 | 0.9109 | 0.8909 | 0.8657 | 0.8483 |
| ricci_gbt_all16 | 0.9324 | 0.9037 | 0.8895 | 0.8804 | 0.8634 |
| hantz_auc_top3_mean | 0.8905 | 0.8929 | 0.8765 | 0.8682 | 0.8381 |
| all16_mean | 0.8738 | 0.8766 | 0.8727 | 0.8227 | 0.7949 |
| direct_greedy_top3 | 0.8584 | 0.8836 | 0.8558 | 0.8365 | 0.8074 |
| direct_exact_top3 | 0.8768 | 0.8736 | 0.8241 | 0.8444 | 0.8148 |
| all16_min | 0.8221 | 0.8186 | 0.8087 | 0.7712 | 0.7480 |
| qubo_exact_top3 | 0.8128 | 0.8219 | 0.8064 | 0.7992 | 0.7627 |
| ricci_gbt_rfe3 | 0.8764 | 0.8533 | 0.8052 | 0.8629 | 0.8266 |
| qubo_greedy_top3 | 0.8077 | 0.8738 | 0.8016 | 0.8195 | 0.7751 |
| single_best | 0.8865 | 0.8470 | 0.7770 | 0.8281 | 0.8053 |
| all16_geometric (primary only) | 0.8760 | NA | NA | 0.8253 | 0.7975 |
| hantz_auc_top3_geometric (primary only) | 0.8886 | NA | NA | 0.8671 | 0.8361 |

## Primary comparison

QUBO exact minus direct greedy BEDROC20: -0.0456.

## Interpretation

Stage 19d is post-hoc PPARG Train-668 development evidence after the Stage 18e technical failure. OOF comparisons reduce resubstitution bias but are not fresh validation. They cannot repair Stage 18e, establish cross-target replication, demonstrate quantum execution, or establish quantum advantage.
