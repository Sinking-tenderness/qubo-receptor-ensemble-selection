# Stage42f BACE1 rank-sensitive pair QUBO

| k | Exact QUBO | Classical | Gap | Exact BEDROC20 | Classical BEDROC20 |
|---:|---|---|---:|---:|---:|
| 1 | BACE1_4I0J_aligned | BACE1_4I0J_aligned | 0 | 0.954334 | 0.954334 |
| 2 | BACE1_3CID_aligned+BACE1_3IVI_aligned | BACE1_3CID_aligned+BACE1_3IVI_aligned | 0 | 0.964183 | 0.964183 |
| 3 | BACE1_3CID_aligned+BACE1_3IVI_aligned+BACE1_4I0J_aligned | BACE1_3CID_aligned+BACE1_3IVI_aligned+BACE1_4I0J_aligned | 0 | 0.973540 | 0.973540 |
| 4 | BACE1_2QP8_aligned+BACE1_3CID_aligned+BACE1_3IVI_aligned+BACE1_4I0J_aligned | BACE1_2QP8_aligned+BACE1_3CID_aligned+BACE1_3IVI_aligned+BACE1_4I0J_aligned | 0 | 0.963209 | 0.963209 |
| 5 | BACE1_2QP8_aligned+BACE1_3CID_aligned+BACE1_3IVH_aligned+BACE1_3IVI_aligned+BACE1_4I0J_aligned | BACE1_2QP8_aligned+BACE1_3CID_aligned+BACE1_3IVH_aligned+BACE1_3IVI_aligned+BACE1_4I0J_aligned | 0 | 0.967551 | 0.967551 |
| 6 | BACE1_2QMF_aligned+BACE1_2QP8_aligned+BACE1_3CID_aligned+BACE1_3IVH_aligned+BACE1_3IVI_aligned+BACE1_4I0J_aligned | BACE1_2QMF_aligned+BACE1_2QP8_aligned+BACE1_3CID_aligned+BACE1_3IVH_aligned+BACE1_3IVI_aligned+BACE1_4I0J_aligned | 0 | 0.967662 | 0.967662 |

## Decision

Rank-sensitive pair QUBO supported: **NO-GO**.

Stage42f is a train-only, outcome-informed BACE1 objective redesign evaluated with scaffold holdouts. A pass would authorize sparse fixed-k QUBO construction and one protected fresh-validation experiment, not quantum hardware or quantum advantage. A failure forbids further BACE1-specific tuning.
