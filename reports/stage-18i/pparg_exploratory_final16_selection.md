# PPARG Exploratory Final-16 Selection

## Decision

Stage 18h completed all 24 reserve redocking jobs and independently passed its
post-hoc recovery gate: 7 of 8 reserve receptors passed, with zero unresolved
warnings and zero pose-integrity failures.

Stage 18e remains a failed confirmatory technical gate at 14 of 24 passing
receptors. Stage 18h does not change that historical result.

## Recovery Selection

The 14 Stage 18e passers were retained. Two receptors were added from the seven
Stage 18h passers by sequential maximum-minimum standardized pocket distance:

1. `PPARG_2P4Y_aligned`: minimum distance `1.2161867494758722`
2. `PPARG_3FUR_aligned`: minimum distance `1.1589696132879803`

RMSD magnitudes, docking affinities, activity labels, validation rows, and test
rows were not used to rank the passing reserves.

## Final Pool

The exploratory final pool contains 16 prepared receptors:

1. `PPARG_2GTK_reference`
2. `PPARG_8B8X_aligned`
3. `PPARG_2HFP_aligned`
4. `PPARG_6AD9_aligned`
5. `PPARG_2Q6S_aligned`
6. `PPARG_3V9V_aligned`
7. `PPARG_3TY0_aligned`
8. `PPARG_2G0H_aligned`
9. `PPARG_4HEE_aligned`
10. `PPARG_5DWL_aligned`
11. `PPARG_6MS7_aligned`
12. `PPARG_2HWR_aligned`
13. `PPARG_2FVJ_aligned`
14. `PPARG_5Y2T_aligned`
15. `PPARG_2P4Y_aligned`
16. `PPARG_3FUR_aligned`

## Next Gate

Prepare the frozen PPARG development panel and generate a three-seed Uni-Dock
matrix for these 16 receptors. All fitting, QUBO selection, greedy and exact
comparators, and literature baselines must remain inside development training
before fresh-validation scores are opened.

This pool and all later PPARG results remain post-hoc exploratory evidence.
