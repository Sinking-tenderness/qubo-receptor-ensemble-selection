# CDK2 Receptor-Ensemble Selection MVP Record

## Purpose

This MVP closes the reproducible evaluation loop for sparse receptor-subset
selection. It is an engineering and methodological milestone, not evidence of
quantum advantage or biological optimality.

## Inputs

- Score matrix: `results/metrics/dude_cdk2_three_receptor_score_matrix.csv`
- Split manifest: `data/processed/dude_cdk2_subset_10a_50d_split_manifest.csv`
- Receptor pool: `1AQ1`, aligned `1HCL`, and aligned `1JVP`
- Ligands: 10 DUD-E actives and 50 DUD-E decoys
- Split: 36 train, 12 validation, 12 test; each validation/test split has 2 actives

## Selection protocol

1. Validate matrix completeness and ligand-ID agreement with the split manifest.
2. Select classical subsets from train only.
3. Build each QUBO from train-only utility, redundancy, and coverage terms.
4. Select QUBO weights on validation only.
5. Evaluate the selected methods on test once, without using test values for selection.

The executable entry point is:

```powershell
conda run -n qubo-receptor-ensemble python scripts/run_receptor_ensemble_mvp.py `
  --matrix results/metrics/dude_cdk2_three_receptor_score_matrix.csv `
  --split-manifest data/processed/dude_cdk2_subset_10a_50d_split_manifest.csv `
  --receptor CDK2_1AQ1_A_prepared CDK2_1HCL_A_aligned_prepared CDK2_1JVP_P_aligned_prepared `
  --output-dir results/metrics/mvp
```

## MVP result

The validation-tuned QUBO selected:

```text
CDK2_1AQ1_A_prepared + CDK2_1JVP_P_aligned_prepared
```

On the held-out test split, this subset achieved ROC-AUC 0.850, PR-AUC 0.700,
BEDROC(alpha=20) 0.842, EF1% 6.0, and EF5% 6.0. The train-selected single
receptor `CDK2_1JVP_P_aligned_prepared` produced the same values on this tiny
test split, so the current result does not demonstrate an ensemble gain.

The validation grid selected zero coverage, overlap, and redundancy weights.
This means the current validation data do not justify the extra QUBO terms;
the result is a useful negative finding, not a reason to remove those terms.

## Generated outputs

- `results/metrics/mvp/receptor_ensemble_mvp.json`
- `results/metrics/mvp/receptor_ensemble_mvp_comparison.csv`
- `results/metrics/mvp/receptor_ensemble_mvp_report.md`

These generated files are local experiment outputs and remain ignored by Git.
The script and this record are the reproducible artifacts committed to the
repository.

## Interpretation and next experiment

The MVP proves that a fixed multi-receptor score matrix can flow through
leakage-aware train/validation/test selection and produce a held-out comparison
between single-receptor, classical subset, all-receptor, and QUBO methods. It
does not yet establish that QUBO improves early enrichment.

The next scientific experiment must expand both axes: a larger, structurally
audited receptor conformer pool and a larger, scaffold-aware ligand benchmark.
The selection protocol and test split should then be frozen before comparing
QUBO with random, clustering, greedy, and train-selected classical baselines.

## Known limitations

- Three receptor conformers are insufficient to test sparse selection at scale.
- Two actives in validation and test make EF1%, BEDROC, and ROC-AUC unstable.
- DUD-E decoys are benchmark decoys, not experimentally confirmed inactives.
- The QUBO currently uses docking-derived utility and score correlation; it is
  not a claim of quantum speedup or quantum advantage.
