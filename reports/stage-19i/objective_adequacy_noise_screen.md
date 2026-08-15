# Stage 19i objective adequacy and noise screen

Post-hoc train-only design review on MK14 and PPARG. No new docking, protected-panel rows, or quantum hardware jobs were used.

## Classical outer-fold results

| Target | Method | Candidate | Mean holdout robust composite | Mean primary |
|---|---|---|---:|---:|
| MK14 | additive_top3 | - | 0.918692 | 0.921664 |
| MK14 | direct_greedy | - | 0.919450 | 0.922534 |
| MK14 | exact_robust_oracle | - | 0.937333 | 0.939831 |
| MK14 | frozen_qubo_objective | consensus_threshold_2 | 0.889109 | 0.888347 |
| MK14 | frozen_qubo_objective | consensus_threshold_2_diverse | 0.904439 | 0.904306 |
| MK14 | frozen_qubo_objective | consensus_threshold_3 | 0.915636 | 0.917939 |
| MK14 | frozen_qubo_objective | union_threshold_1 | 0.904192 | 0.907688 |
| PPARG | additive_top3 | - | 0.890256 | 0.905514 |
| PPARG | direct_greedy | - | 0.841859 | 0.841644 |
| PPARG | exact_robust_oracle | - | 0.841859 | 0.841644 |
| PPARG | frozen_qubo_objective | consensus_threshold_2 | 0.811449 | 0.806527 |
| PPARG | frozen_qubo_objective | consensus_threshold_2_diverse | 0.819026 | 0.816553 |
| PPARG | frozen_qubo_objective | consensus_threshold_3 | 0.799590 | 0.798794 |
| PPARG | frozen_qubo_objective | union_threshold_1 | 0.826056 | 0.822267 |

## Hardware-readiness checks

| Candidate | Classical gate | Hardware gate | Ready |
|---|---:|---:|---:|
| union_threshold_1 | False | False | False |
| consensus_threshold_2 | False | False | False |
| consensus_threshold_3 | False | False | False |
| consensus_threshold_2_diverse | False | False | False |

- Ready candidates: `none`
- Next gate: `redesign_objective_or_reframe_quantum_application_before_hardware`

Stage 19i is a post-hoc train-only design review over MK14 and PPARG. It adds no docking and reads no fresh-validation, locked-test, or BACE1 rows. A passing candidate would be a hardware-readiness signal only; a failure means the candidate is not ready for hardware, not that all quantum approaches are impossible.
