# Stage42d BACE1 large-pool QUBO screen

The Stage37 robust functional objective was reused without BACE1-specific weight tuning.

| Method | k | Objective | Robust BEDROC20 | Subset |
|---|---:|---:|---:|---|
| exact_qubo_objective_optimum | 6 | 0.28984962 | 0.871603 | BACE1_2QMG_aligned+BACE1_2QP8_aligned+BACE1_3IVH_aligned+BACE1_3L5E_aligned+BACE1_4DJV_aligned+BACE1_4HZT_aligned |
| beam64_multistart_add_drop_swap | 6 | 0.28984962 | 0.871603 | BACE1_2QMG_aligned+BACE1_2QP8_aligned+BACE1_3IVH_aligned+BACE1_3L5E_aligned+BACE1_4DJV_aligned+BACE1_4HZT_aligned |
| direct_greedy | 6 | 0.28233083 | 0.926244 | BACE1_2QP8_aligned+BACE1_3CIC_aligned+BACE1_3IVI_aligned+BACE1_3L5E_aligned+BACE1_4I0J_aligned+BACE1_4RRS_aligned |
| exact_additive_singleton | 6 | 0.23007519 | 0.968044 | BACE1_2QMD_aligned+BACE1_2QP8_aligned+BACE1_3CID_aligned+BACE1_3IVH_aligned+BACE1_3IVI_aligned+BACE1_4I0J_aligned |
| random_matched_budget_best | 6 | 0.28082707 | 0.881235 | BACE1_2QMF_aligned+BACE1_2QMG_aligned+BACE1_3IVH_aligned+BACE1_3L5E_aligned+BACE1_4DJV_aligned+BACE1_4I0J_aligned |
| best_single_receptor | 1 | 0.09768170 | 0.954334 | BACE1_4I0J_aligned |
| all_34_receptors | 34 | 0.15952381 | 0.827177 | BACE1_2QMD_aligned+BACE1_2QMF_aligned+BACE1_2QMG_aligned+BACE1_2QP8_aligned+BACE1_3CIB_aligned+BACE1_3CIC_aligned+BACE1_3CID_aligned+BACE1_3IVH_aligned+BACE1_3IVI_aligned+BACE1_3KN0_aligned+BACE1_3L59_aligned+BACE1_3L5E_aligned+BACE1_4DJU_aligned+BACE1_4DJV_aligned+BACE1_4DJW_aligned+BACE1_4DJX_aligned+BACE1_4DJY_aligned+BACE1_4FS4_aligned+BACE1_4H3F_aligned+BACE1_4H3G_aligned+BACE1_4H3I_aligned+BACE1_4H3J_aligned+BACE1_4HA5_aligned+BACE1_4HZT_aligned+BACE1_4I0F_aligned+BACE1_4I0G_aligned+BACE1_4I0J_aligned+BACE1_4I0Z_aligned+BACE1_4I10_aligned+BACE1_4I11_aligned+BACE1_4RRN_aligned+BACE1_4RRO_aligned+BACE1_4RRS_aligned+BACE1_6PZ4_aligned |

## Decision

Frozen objective support gate: **NO-GO**.

Stage42d uses only the outcome-informed BACE1 Train-266 matrix and a functional objective frozen before BACE1 scores existed. A pass would justify sparse auxiliary-QUBO construction and one protected fresh-validation test; it would not establish independent multi-protein replication, quantum execution, or quantum advantage. A failure does not permit BACE1-specific weight tuning.
