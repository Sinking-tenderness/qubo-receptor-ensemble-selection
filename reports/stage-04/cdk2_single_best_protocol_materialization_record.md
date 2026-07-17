# Stage 4 CDK2 Single-Best Protocol Materialization Record

## Status

- Run date: 2026-07-17
- Status: `single_best_protocol_materialized_test_locked`
- Upstream protocol: `single_best_fail_closed_v1`
- Selected receptor: `CDK2_AF2_MD2NS_C06_F077`
- Receptor source: 2 ns AF2 MD cluster medoid, frame 77
- Receptor chain: A
- Receptors compared: 16
- Development ligands: 160 (80 active, 80 decoy)
- Locked-test ligands: 40 (20 active, 20 decoy)
- Locked-test score cells read: 0
- Test metrics computed: no
- Test release authorized: no

This step converts the reliable fallback method into a concrete receptor and
file identity. It does not evaluate the locked test and does not change the
negative QUBO reliability decision.

## Selection Rule

The selected method was fixed before this refit by
`stage04_cdk2_reliable_protocol_selector.json`.

1. Use only train plus validation ligands from the scaffold split.
2. Rank each receptor independently with across-seed median Vina scores.
3. Select the largest BEDROC20 value.
4. Break ties by PR-AUC, then ROC-AUC, then receptor ID.
5. Use the across-seed minimum matrix only as a sensitivity audit.
6. Fix the receptor ID and PDBQT SHA-256 before any test evaluation.

The development matrix IDs exactly matched the 160 development IDs. Both the
primary and sensitivity matrices had zero overlap with the 40 locked-test IDs.

## Selected Receptor

| Field | Value |
|---|---|
| Receptor ID | `CDK2_AF2_MD2NS_C06_F077` |
| Source type | `md_cluster_medoid` |
| Aligned heavy PDB | `results/runs/stage03_cdk2_af2_md_cuda/medoid_receptors/aligned_heavy/CDK2_AF2_MD2NS_C06_F077_to_1AQ1_A_heavy.pdb` |
| PDB SHA-256 | `8F5025BF26F1F2B95A4EFD315FE3E872E2778F4428A20AA6CBFA2B4D2349DBE5` |
| Prepared PDBQT | `results/runs/stage03_cdk2_af2_md_cuda/medoid_receptors/prepared/CDK2_AF2_MD2NS_C06_F077/CDK2_AF2_MD2NS_C06_F077_receptor.pdbqt` |
| PDBQT SHA-256 | `ECFC7BF4ADFD56E78EAE5F3A121C419612457B0EABA52DA7928627F53C0FE84E` |
| PDBQT atoms | 2,912 |
| Hydrogen-like atoms | 514 |
| HETATM records | 0 |
| Pocket residues | 21 |
| Charge range | -0.549 to 0.345 |
| AutoDock atom types | A, C, HD, N, NA, OA, SA |

The materializer verified the PDB and PDBQT bytes against the hash-pinned
expanded receptor manifest. A receptor with the same name but different bytes
cannot silently replace this artifact.

## Development Refit Metrics

| Matrix | Receptor rank | ROC-AUC | PR-AUC | BEDROC20 | EF1% | EF5% | EF10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Median primary | 1 / 16 | 0.69195 | 0.75650 | 0.95313 | 2.000 | 2.000 | 1.875 |
| Minimum sensitivity | 1 / 16 | 0.68719 | 0.75114 | 0.94624 | 2.000 | 2.000 | 1.875 |

These are full-development refit metrics. They describe why this receptor was
fixed, but they are not held-out performance estimates.

The five highest primary BEDROC20 receptors were:

| Rank | Receptor | BEDROC20 | PR-AUC | ROC-AUC |
|---:|---|---:|---:|---:|
| 1 | `CDK2_AF2_MD2NS_C06_F077` | 0.95313 | 0.75650 | 0.69195 |
| 2 | `CDK2_1AQ1_reference` | 0.94349 | 0.74887 | 0.69555 |
| 3 | `CDK2_3RKB_aligned` | 0.93318 | 0.73043 | 0.69141 |
| 4 | `CDK2_1JVP_aligned` | 0.89306 | 0.72895 | 0.69523 |
| 5 | `CDK2_AF_P24941_F1_v6` | 0.89180 | 0.72606 | 0.70203 |

## Identity Stability

The selected receptor was also chosen as the fold-specific single-best
receptor in 14 of 20 outer folds across five scaffold-fold seeds (70%). The
remaining selections were 1AQ1 in four folds, 1Y8Y in one fold, and 3RKB in one
fold.

This frequency is descriptive evidence that the identity is not driven by one
development split. It was not introduced as a post-hoc acceptance threshold.

## Interpretation Boundary

The protocol is now fixed enough for a future one-time test evaluation. Such an
evaluation must use only the recorded receptor PDBQT, the already fixed docking
settings and score aggregation, and the existing 40-ligand test split. The
receptor, metric, or protocol may not be changed after test results are seen.

This result does not demonstrate biological superiority, binding affinity
accuracy, or quantum advantage. It provides a reproducible classical fallback
and a clean benchmark that future QUBO candidates must beat on independent
evidence.

## Verification

- Materializer unit and integration tests: 3 passed.
- Full repository tests after implementation: 157 passed, 1 skipped.
- Primary and sensitivity locked-test overlap: 0.
- Repeated-CV source runs: 5 / 5 hash verified.
- Outer-fold rows used for stability: 20 / 20.
- Every repeated-CV source summary retained `scores_evaluated = false`.

## Output Hashes

| Output | SHA-256 |
|---|---|
| `protocol.json` | `E0FB629C37B5AE8EEBC97E3CBBF3E63CE8CC2A964B54053EE8076D1E2528C913` |
| `receptor_ranking.csv` | `87C313530A3EA9E47BE53CE5249CCE090FAF3AE7F20BEF8C9156E14434653C9E` |

Machine-readable outputs:

- `results/runs/stage04_cdk2_single_best_protocol_materialization/protocol.json`
- `results/runs/stage04_cdk2_single_best_protocol_materialization/receptor_ranking.csv`
