# Stage 4 CDK2 Reliable Protocol Selection Record

## Decision

- Run date: 2026-07-17
- Selector status: `fallback_selected_no_reliable_qubo_candidate`
- Candidate count: 5
- Reliable QUBO candidate count: 0
- Active protocol: `single_best_fail_closed_v1`
- Method: `single_best`
- Aggregation: `min_score`
- Selection metric: `bedroc_alpha_20`
- Selection data: development only
- Locked-test scores evaluated: no

The reliable result is a fail-closed selection protocol, not a claim that a
QUBO subset has already improved virtual screening. None of the evaluated QUBO
structures passed every repeated-CV criterion, so the selector retained the
single-receptor baseline and did not authorize test release.

## Data Boundary

The receptor pool contains 16 aligned CDK2 conformers. Model selection used 160
development ligands (80 active and 80 decoy) in scaffold-disjoint CV. The 40
test ligands (20 active and 20 decoy) remained excluded from score parsing,
fitting, ranking, metric calculation, and bootstrap resampling.

The primary matrix contains across-seed median Vina scores. The sensitivity
matrix contains across-seed minimum scores. All successful repeated gates were
independently audited before their summaries were accepted by the selector.

## Reliability Gate

A QUBO candidate had to pass every check below. The thresholds were fixed in
`configs/stage04_cdk2_reliable_protocol_selector.json` before the final
selector run.

| Check | Required |
|---|---:|
| Feasible repeated gates | 5 / 5 |
| Repeats with positive primary BEDROC20 delta | at least 4 / 5 |
| Mean primary BEDROC20 delta | at least +0.02000 |
| Mean primary ROC-AUC delta | at least 0.00000 |
| Mean primary PR-AUC delta | at least 0.00000 |
| Mean sensitivity BEDROC20 delta | at least 0.00000 |
| Aggregate paired-bootstrap BEDROC20 CI95 lower bound | at least 0.00000 |

The comparator for every delta was the fold-specific `single_best` method.
Repeated feasibility was counted against all five requested seeds, so a failed
or infeasible run could not silently disappear from the denominator.

## Candidate Results

| Candidate | Feasible | Positive BEDROC20 | Mean BEDROC20 delta | Mean ROC-AUC delta | Mean PR-AUC delta | Sensitivity BEDROC20 delta | Bootstrap CI95 low | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Strict two-core plus one | 1/5 | 1/5 | +0.01429 | +0.02109 | +0.01149 | +0.00942 | -0.11675 | No |
| One-core plus two | 5/5 | 1/5 | -0.03051 | +0.00670 | -0.00912 | -0.03452 | -0.09545 | No |
| Unconstrained coverage | 5/5 | 2/5 | +0.01605 | +0.00523 | +0.00246 | +0.01213 | -0.04478 | No |
| Fixed-2 mean coverage | 5/5 | 2/5 | -0.01161 | -0.01895 | -0.02507 | -0.01501 | -0.18418 | No |
| Fixed-3 mean coverage | 5/5 | 2/5 | +0.01811 | +0.02580 | +0.02082 | +0.01160 | -0.09326 | No |

The fixed-3 mean-aggregation structure was the strongest QUBO candidate by
mean global-ranking deltas and included a favorable individual seed. It still
failed repeat consistency, the +0.02 mean BEDROC20 requirement, and bootstrap
support. Promoting it would therefore select an unstable development result.

The strict two-core structure was infeasible in four repeats because two
frequency-qualified core receptors were not consistently available. Relaxing
the structure to one core plus two residual receptors restored feasibility but
made the average early-enrichment result worse than single-best. The
unconstrained and fixed-size coverage variants produced mixed fold-seed signs,
showing that the apparent benefit depends strongly on scaffold partitioning.

## Selected Protocol

`single_best_fail_closed_v1` means:

1. Use only the hash-pinned 160-ligand development matrices.
2. Compare individual receptors using scaffold-disjoint development CV.
3. Select by BEDROC20, with PR-AUC and ROC-AUC used only as tie-breakers.
4. Fit the chosen single-receptor rule on all development ligands only after
   the method has been selected.
5. Fix the receptor and protocol before any locked-test score is evaluated.
6. Do not release a QUBO subset unless a later preregistered candidate passes
   every repeated-CV reliability check.

This is an executable model-selection rule, but the current selector is not a
test-release command. A separate pre-test materialization step must record the
specific full-development receptor refit and its hashes before a one-time test
evaluation can be authorized.

## Interpretation

The experiments establish three useful boundaries:

- A feasible QUBO formulation is not automatically a reliable screening
  protocol.
- Small positive mean deltas are insufficient when fold-seed direction changes
  and the paired-bootstrap interval includes harm.
- Further tuning on the same 160 ligands would increase development-set
  overfitting risk. A new QUBO objective should be preregistered and preferably
  evaluated with additional independent development data before reconsidering
  the locked test.

The result does not establish biological superiority, docking-score accuracy,
or quantum advantage. It does establish a reproducible stopping rule that
prevents an unstable QUBO result from being promoted.

## Verification

- Final selector rerun completed with `candidate_pass_count = 0`.
- Final selector rerun reported `test_evaluated = false`.
- Full repository tests in `qubo-receptor-ensemble`: 154 passed, 1 skipped.
- Every successful repeated gate required an independent audit to pass.
- Candidate summaries and repeated output tables were verified by SHA-256
  before metric recomputation.

## Output Hashes

| Output | SHA-256 |
|---|---|
| Reliable selector `summary.json` | `C8B96818AABB39F9C079E9C3C47DDE1E9EEB0AC909B27204B3E46B3793781AAB` |
| Reliable selector `decision_table.csv` | `1BAF518F14F0249304E7C05E6DAE70B06DCA0FB6F2D0C497A7F8DAE62D656C03` |
| Strict core-plus-one repeated summary | `9F32D4A3008384495C6A56F4479334F085EACFFC9C98001E67DCA6EC271040EA` |
| Core-plus-two repeated summary | `E20E735F709570B47FF7937FA2B9B3C9DCC681C4806947AE137D24D126122FB8` |
| Fixed-2 mean repeated summary | `602E17BA41AAE854EB9FABFA057012E8539EFBD86FF8767537AAC01B33705548` |
| Fixed-3 mean repeated summary | `660BA82E4F2D21810D3BA653B9E94BD0F420C2983A4FE59C28753099F9FAE62A` |

Primary machine-readable outputs:

- `results/runs/stage04_cdk2_reliable_protocol_selector/summary.json`
- `results/runs/stage04_cdk2_reliable_protocol_selector/decision_table.csv`
