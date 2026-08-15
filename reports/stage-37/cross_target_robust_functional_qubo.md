# Stage37 cross-target robust functional QUBO screen

This frozen train-only screen compares the exact optimum of a robust functional set objective with a beam-64, multi-start, add/drop/swap classical search.
No fresh-validation or test row was read.

| Target | Fold | Exact subset | Classical subset | Train gap | Holdout objective delta | Holdout BEDROC delta |
|---|---:|---|---|---:|---:|---:|
| MK14 | 0 | MK14_3KQ7_aligned+MK14_2BAJ_aligned+MK14_4F9W_aligned+MK14_3ZSI_aligned+MK14_3ZSG_aligned+MK14_4AAC_aligned | MK14_3KQ7_aligned+MK14_2BAJ_aligned+MK14_4F9W_aligned+MK14_3ZSI_aligned+MK14_3ZSG_aligned+MK14_4AAC_aligned | 0.00000000 | +0.000000 | +0.000000 |
| MK14 | 1 | MK14_2BAJ_aligned+MK14_4F9W_aligned+MK14_3ZSI_aligned+MK14_3BV2_aligned+MK14_4FA2_aligned+MK14_3ITZ_aligned | MK14_2BAJ_aligned+MK14_4F9W_aligned+MK14_3ZSI_aligned+MK14_3BV2_aligned+MK14_4FA2_aligned+MK14_3ITZ_aligned | 0.00000000 | +0.000000 | +0.000000 |
| MK14 | 2 | MK14_2BAJ_aligned+MK14_4F9W_aligned+MK14_3ZSI_aligned+MK14_3BV2_aligned+MK14_4FA2_aligned+MK14_3ITZ_aligned | MK14_2BAJ_aligned+MK14_4F9W_aligned+MK14_3ZSI_aligned+MK14_3BV2_aligned+MK14_4FA2_aligned+MK14_3ITZ_aligned | 0.00000000 | +0.000000 | +0.000000 |
| MK14 | 3 | MK14_2BAJ_aligned+MK14_3ZSI_aligned+MK14_3BV2_aligned+MK14_4FA2_aligned+MK14_4AAC_aligned+MK14_3ITZ_aligned | MK14_2BAJ_aligned+MK14_3ZSI_aligned+MK14_3BV2_aligned+MK14_4FA2_aligned+MK14_4AAC_aligned+MK14_3ITZ_aligned | 0.00000000 | +0.000000 | +0.000000 |
| PPARG | 0 | PPARG_2GTK_reference+PPARG_2Q6S_aligned+PPARG_2G0H_aligned+PPARG_5DWL_aligned+PPARG_5Y2T_aligned+PPARG_3FUR_aligned | PPARG_2GTK_reference+PPARG_2Q6S_aligned+PPARG_2G0H_aligned+PPARG_5DWL_aligned+PPARG_5Y2T_aligned+PPARG_3FUR_aligned | 0.00000000 | +0.000000 | +0.000000 |
| PPARG | 1 | PPARG_2Q6S_aligned+PPARG_3TY0_aligned+PPARG_2G0H_aligned+PPARG_5DWL_aligned+PPARG_5Y2T_aligned+PPARG_3FUR_aligned | PPARG_2Q6S_aligned+PPARG_3TY0_aligned+PPARG_2G0H_aligned+PPARG_5DWL_aligned+PPARG_5Y2T_aligned+PPARG_3FUR_aligned | 0.00000000 | +0.000000 | +0.000000 |
| PPARG | 2 | PPARG_2Q6S_aligned+PPARG_2G0H_aligned+PPARG_5DWL_aligned+PPARG_5Y2T_aligned+PPARG_2P4Y_aligned+PPARG_3FUR_aligned | PPARG_2Q6S_aligned+PPARG_2G0H_aligned+PPARG_5DWL_aligned+PPARG_5Y2T_aligned+PPARG_2P4Y_aligned+PPARG_3FUR_aligned | 0.00000000 | +0.000000 | +0.000000 |
| PPARG | 3 | PPARG_2Q6S_aligned+PPARG_2G0H_aligned+PPARG_5DWL_aligned+PPARG_6MS7_aligned+PPARG_5Y2T_aligned+PPARG_3FUR_aligned | PPARG_2Q6S_aligned+PPARG_2G0H_aligned+PPARG_5DWL_aligned+PPARG_6MS7_aligned+PPARG_5Y2T_aligned+PPARG_3FUR_aligned | 0.00000000 | +0.000000 | +0.000000 |

## Decision

- Functional objective support gate: **NO-GO**.
- Sparse auxiliary-QUBO encoding authorized: `False`.
- Quantum hardware authorized: `False`.

## Boundary

Stage37 is post-hoc objective development using only frozen MK14 and PPARG training rows. Scaffold holdouts test transfer within those development sets but are not independent protein validation. A pass would support sparse auxiliary-QUBO construction, not quantum hardware, quantum advantage, new docking, or alteration of any prior failed confirmation gate. A failure freezes this objective as unsupported and forbids further weight tuning on these same outcomes.
