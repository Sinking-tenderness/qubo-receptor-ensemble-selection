# Stage36 PPARG consensus-objective landscape

Each cell exhaustively enumerates 4^8 = 65,536 feasible selections.

| Objective | Passing cohorts | Max greedy gap | Max strict local optima | Min optimum basin | Gate |
|---|---:|---:|---:|---:|---|
| coarse_consensus_fine_diversity | 0/3 | 0 | 1 | 1.000000 | NO-GO |
| mid_consensus_fine_diversity | 0/3 | 0 | 1 | 1.000000 | NO-GO |
| frustrated_hierarchical_pair | 0/3 | 0 | 1 | 1.000000 | NO-GO |
| geometry_frustrated_pair | 0/3 | 0 | 3 | 0.722519 | NO-GO |
| coarse_triple_supported_portfolio | 0/3 | 0 | 2 | 0.458191 | NO-GO |
| hierarchical_double_consensus | 0/3 | 0 | 1 | 1.000000 | NO-GO |
| threshold_consensus_control | 0/3 | 0 | 0 | 1.000000 | NO-GO |
| smooth_stage30_control | 0/3 | 0 | 1 | 1.000000 | NO-GO |

Selected Stage37 objective: **NONE**.

Stage36 can identify a structure-only objective with exact evidence of nontrivial local optima and a reproducible strong-greedy gap on three frozen PPARG MD cohorts. Passing authorizes a separately frozen sparse-QUBO encoding and solver-scaling study. It cannot establish docking enrichment, biological superiority, quantum speedup, quantum advantage, or hardware readiness.
