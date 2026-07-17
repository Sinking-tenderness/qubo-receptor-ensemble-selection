# Stage 4 CDK2 Single-Best Locked-Test Evaluation Record

## Release Status

- Evaluation date: 2026-07-17
- Authorization: project owner approved
- Authorization ID: `stage04-cdk2-locked-test-release-20260717-v1`
- Preregistration commit: `9be0d0f`
- Release status: `locked_test_evaluated_once`
- Release consumed: yes
- Repeat or overwrite allowed: no
- Fixed protocol: `single_best_fail_closed_v1`
- Fixed receptor: `CDK2_AF2_MD2NS_C06_F077`
- Test ligands: 40 (20 active, 20 decoy)
- Receptor score columns evaluated: 1

The release script, receptor ID, receptor PDBQT hash, source matrices, ranking
rule, metrics, bootstrap settings, and absence of an acceptance threshold were
committed before test scores were parsed.

## Fixed Artifact

| Field | Value |
|---|---|
| Receptor PDBQT | `results/runs/stage03_cdk2_af2_md_cuda/medoid_receptors/prepared/CDK2_AF2_MD2NS_C06_F077/CDK2_AF2_MD2NS_C06_F077_receptor.pdbqt` |
| PDBQT SHA-256 | `ECFC7BF4ADFD56E78EAE5F3A121C419612457B0EABA52DA7928627F53C0FE84E` |
| Primary score source | across-seed median, exhaustiveness 32 |
| Sensitivity score source | across-seed minimum, exhaustiveness 32 |
| Ranking order | docking score ascending, then ligand ID ascending |
| Bootstrap | 5,000 stratified active/decoy resamples |
| Bootstrap seed | 20261031 (primary), 20261032 (sensitivity) |

No receptor reselection, QUBO fitting, threshold tuning, or comparison against
another receptor was performed after release.

## Test Metrics

| Matrix | ROC-AUC | PR-AUC | BEDROC20 | EF1% | EF5% | EF10% | Top10 active |
|---|---:|---:|---:|---:|---:|---:|---:|
| Median primary | 0.80625 | 0.84763 | 0.99385 | 2.000 | 2.000 | 2.000 | 9 / 10 |
| Minimum sensitivity | 0.81250 | 0.84696 | 0.99322 | 2.000 | 2.000 | 2.000 | 9 / 10 |

Because the test prevalence is 50%, an enrichment factor of 2.0 is the maximum
possible value. The first ligand, first two ligands, and first four ligands were
all active, producing maximum EF1%, EF5%, and EF10%, respectively.

## Uncertainty

The intervals below came from the preregistered stratified bootstrap.

| Matrix | Metric | Point estimate | Bootstrap mean | 95% interval |
|---|---|---:|---:|---:|
| Primary | ROC-AUC | 0.80625 | 0.80464 | [0.65500, 0.92250] |
| Primary | PR-AUC | 0.84763 | 0.84652 | [0.72761, 0.93953] |
| Primary | BEDROC20 | 0.99385 | 0.98974 | [0.94780, 0.99955] |
| Sensitivity | ROC-AUC | 0.81250 | 0.81169 | [0.66500, 0.92750] |
| Sensitivity | PR-AUC | 0.84696 | 0.84659 | [0.72697, 0.93855] |
| Sensitivity | BEDROC20 | 0.99322 | 0.98881 | [0.94318, 0.99952] |

The early-enrichment result is stable across median and minimum aggregation.
The ROC-AUC interval remains wide because only 40 ligands were available.

## Top Ten

| Rank | Ligand | Label | Median docking score |
|---:|---|---|---:|
| 1 | `CDK2_A0092` | active | -10.180 |
| 2 | `CDK2_A0099` | active | -9.170 |
| 3 | `CDK2_A0033` | active | -9.000 |
| 4 | `CDK2_A0003` | active | -8.785 |
| 5 | `CDK2_A0017` | active | -8.745 |
| 6 | `CDK2_A0067` | active | -8.682 |
| 7 | `CDK2_A0009` | active | -8.638 |
| 8 | `CDK2_A0087` | active | -8.533 |
| 9 | `CDK2_A0085` | active | -8.418 |
| 10 | `CDK2_D0017` | decoy | -8.401 |

`CDK2_D0017` is the only decoy in the first ten. It should be retained as a
high-scoring decoy diagnostic, not deleted or reclassified after seeing the
test result.

## Independent Audit

`scripts/audit_locked_test_release.py` read only the released ranking CSVs and
the split manifest. It did not reopen the source score matrices or evaluate a
second receptor.

- Test IDs reproduced: 40 / 40.
- Labels reproduced: 20 active and 20 decoy.
- Rank sequence and deterministic score/ID ordering: valid.
- Primary and sensitivity metrics reproduced: yes.
- Maximum absolute metric difference: 0.0.
- Top10 IDs and labels reproduced: yes.
- Bootstrap record count and bounds: valid.
- Release marker consumed and rerun disabled: verified.

## Interpretation

The fixed C06 single-receptor protocol generalized favorably to the independent
40-ligand scaffold test split, especially for early enrichment. This is valid
evidence that the classical fallback protocol is useful for the current CDK2
DUD-E benchmark.

It is not evidence that Vina scores equal binding free energies, that DUD-E
decoys are experimentally inactive, or that the result generalizes to other
targets or prospective compounds. The small test set and 50% artificial class
balance also differ from production virtual-screening prevalence.

Most importantly, this result does not validate QUBO or demonstrate quantum
advantage. No QUBO candidate passed the development reliability gate, so the
tested protocol is the preregistered classical single-receptor fallback.

The 40 test ligands are now consumed and may not be used for future receptor
selection, QUBO weight tuning, acceptance-threshold design, or model revision.
Any further method development requires new external validation data or a new
target-level benchmark.

## Verification

- Preregistration tests before release: 160 passed, 1 skipped.
- Locked-test release execution count: 1.
- Post-release focused audit tests: 4 passed.
- Final full repository tests: 161 passed, 1 skipped.
- Independent audit status: `ok`.
- Metrics reproduced with maximum difference: 0.0.

## Output Hashes

| Output | SHA-256 |
|---|---|
| Release `summary.json` | `9EA3417B458DD55CB4AB57A9EE7EF803117CD100F307916D5BFF17346BF70EA2` |
| `release_marker.json` | `8D51FFE3F5C95172B04B6E91EE92F0B9D784CB6EDFB7F701D9477168FD2DDB03` |
| Primary test rankings | `CDFF90BF8EF4D73FF0BB2C77F7B56905F8A48AB0E12305C6A1D4214E596B3165` |
| Sensitivity test rankings | `3E513937667B92D84AE5D55FAE13A6870FE1060400F3DF36AB7B22B786CCA81E` |
| Independent audit | `E05F8D3E428F8E978760CA3ABC30690249664FE2CE7EFDA812B72D98F7E39138` |

Machine-readable outputs are under
`results/runs/stage04_cdk2_single_best_locked_test_release/`.
