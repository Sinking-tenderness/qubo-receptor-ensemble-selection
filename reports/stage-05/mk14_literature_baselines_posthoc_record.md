# MAPK14 Post Hoc Literature Baseline Record

## Scope and evidence timing

These literature-family baselines were specified after the fresh-validation
results were already available. The implementation fits and selects using
Train-696 only, but the comparison is post hoc and cannot modify the frozen
primary gate or support a new confirmatory significance claim. The locked test
was not read.

## Primary fresh-validation comparison

| Rank | Method | Timing | Receptors | BEDROC20 | ROC-AUC | PR-AUC | Delta vs QUBO |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | nested_greedy_final | preregistered_primary | 3 | 0.5509 | 0.8360 | 0.4265 | +0.0000 |
| 2 | pair_synergy_qubo | preregistered_primary | 3 | 0.5509 | 0.8360 | 0.4265 | +0.0000 |
| 3 | edock_rf_all5 | posthoc_train_only_fit | 5 | 0.5377 | 0.8529 | 0.3903 | -0.0131 |
| 4 | xgboost_all5 | preregistered_supplementary | 5 | 0.5361 | 0.8376 | 0.3868 | -0.0147 |
| 5 | nested_exhaustive_final | preregistered_primary | 3 | 0.5256 | 0.8323 | 0.3996 | -0.0252 |
| 6 | xgboost_budget3 | preregistered_supplementary | 3 | 0.5106 | 0.8352 | 0.3643 | -0.0402 |
| 7 | ricci_gbt_all5 | posthoc_train_only_fit | 5 | 0.5098 | 0.8258 | 0.3386 | -0.0411 |
| 8 | ricci_lr_all5 | posthoc_train_only_fit | 5 | 0.4825 | 0.8237 | 0.3435 | -0.0683 |
| 9 | hantz_auc_top3_min | posthoc_train_only_fit | 3 | 0.4792 | 0.8083 | 0.3435 | -0.0717 |
| 10 | hantz_auc_top3_mean | posthoc_train_only_fit | 3 | 0.4767 | 0.8233 | 0.3356 | -0.0742 |
| 11 | hantz_auc_top3_geometric | posthoc_train_only_fit | 3 | 0.4719 | 0.8228 | 0.3278 | -0.0790 |
| 12 | consensus_mean_all5 | posthoc_train_only_fit | 5 | 0.4634 | 0.8067 | 0.3428 | -0.0875 |
| 13 | consensus_geometric_all5 | posthoc_train_only_fit | 5 | 0.4620 | 0.8072 | 0.3395 | -0.0889 |
| 14 | matched_linear_top_k | preregistered_primary | 3 | 0.4585 | 0.8167 | 0.2700 | -0.0924 |
| 15 | edock_rf_rfe3 | posthoc_train_only_fit | 3 | 0.4548 | 0.8139 | 0.2920 | -0.0961 |
| 16 | ricci_gbt_rfe3 | posthoc_train_only_fit | 3 | 0.4483 | 0.7958 | 0.3087 | -0.1025 |
| 17 | consensus_min_all5 | posthoc_train_only_fit | 5 | 0.4138 | 0.7805 | 0.3134 | -0.1370 |
| 18 | single_best | preregistered_primary | 1 | 0.3760 | 0.7258 | 0.2632 | -0.1748 |

## Interpretation

The best newly added baseline was `edock_rf_all5` with BEDROC20=0.5377. The frozen QUBO/greedy ranking remained at BEDROC20=0.5509, a difference of -0.0131 for the new method minus QUBO.

An exploratory, post hoc split-group bootstrap for QUBO minus the best
new RF baseline gave a 95% interval of [-0.0582, 0.0872]. Because this interval
crosses zero and the comparator was inspected post hoc, it does not support
a claim that QUBO is statistically superior to RF.

The result tests whether standard linear, tree, feature-selection, and
consensus families explain the validation performance. It does not establish
quantum advantage, and method choice must not be revised using this validation
table before the locked-test protocol is frozen.

## Fixed literature mapping

- Ricci-Lopez et al. (JCIM 2021): LR, GBT, RFE, MIN, AVG, and GEO families.
- Chandak et al. / EDock-ML: random-forest score-matrix classifier family.
- Hantz and Lindert (JCIM 2022): top-three receptors by train-only singleton ROC-AUC.
- Swift et al. (JCIM 2016): already represented by the frozen exhaustive, greedy, and linear top-k rows.

Configuration: `stage05-mk14-literature-baselines-posthoc-20260724-v1`.
