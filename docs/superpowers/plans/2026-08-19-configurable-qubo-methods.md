# Configurable QUBO Methods Implementation Plan

> **For implementation:** execute this plan in the current repository. Keep
> existing single-method Stage102A configs working while adding method
> comparison and historical capability gating.

**Goal:** Make BEDROC20 the default selection objective, expose the BEDROC
alpha parameter, register historical QUBO objective/encoding families, and
run compatible methods side by side from one aggregated docking matrix.

**Architecture:** Keep `Problem` and `SolverResult` as the solver-independent
contracts. Add a method registry that builds a `Problem` formulation and
declares input capabilities. Preserve `strategy: qubo` and
`strategy: normalized_qubo` as compatibility aliases. Add a comparison path
at the experiment stage boundary so preparation, docking, and aggregation are
shared.

**Default policy:** `utility_metric=bedroc`, `bedroc_alpha=20.0`, and
`bedroc_alpha_20` is the default selection/evaluation primary metric.
ROC-AUC remains an explicit secondary diagnostic.

## Task 1: Lock configuration and metric defaults

**Files:**
- Modify: `src/qubo_receptor_ensemble/full_workflow.py`
- Modify: `src/qubo_receptor_ensemble/qubo.py`
- Modify: `src/qubo_receptor_ensemble/solvers.py`
- Modify: `src/qubo_receptor_ensemble/k_selection.py`
- Test: `tests/test_full_experiment_config.py`
- Test: `tests/test_solver_adapters.py`
- Test: `tests/test_pipeline_foundation.py`

- [ ] Add validation for `problem.utility_metric` and numeric positive
  `problem.bedroc_alpha`; default omitted values to BEDROC20.
- [ ] Pass `bedroc_alpha` through `build_qubo` and serialize it in the
  formulation.
- [ ] Make adaptive-k selection default to the configured BEDROC metric.
- [ ] Add failing tests proving omitted utility settings resolve to BEDROC20
  and explicit ROC-AUC remains supported.
- [ ] Run the focused tests and confirm the new tests fail before implementation.
- [ ] Implement the smallest changes to make the focused tests pass.

## Task 2: Add a method registry and capability contract

**Files:**
- Create: `src/qubo_receptor_ensemble/methods.py`
- Modify: `src/qubo_receptor_ensemble/solvers.py`
- Modify: `src/qubo_receptor_ensemble/full_workflow.py`
- Test: `tests/test_qubo_methods.py`

- [ ] Define immutable method metadata: `method_id`, provenance, formulation
  kind, required inputs, supported backends, and default parameters.
- [ ] Register score-matrix methods that can share the current pipeline:
  `basic_utility`, `pair_utility`, `pair_synergy`,
  `rank_sensitive_pair`, `uncertainty_shrunk`, `normalized_coverage`,
  `auxiliary_coverage`, and `rankbin_bedroc20`.
- [ ] Register structural, HUBO, encoding, and hardware families with explicit
  requirements and no silent score-only fallback.
- [ ] Add capability validation that returns `ready` or
  `unsupported_for_input` with missing artifact names.
- [ ] Add tests for registry lookup, unknown methods, default parameters, and
  missing capabilities.

## Task 3: Implement compatible historical objective builders

**Files:**
- Modify: `src/qubo_receptor_ensemble/qubo.py`
- Create: `src/qubo_receptor_ensemble/method_formulations.py`
- Modify: `src/qubo_receptor_ensemble/solvers.py`
- Test: `tests/test_qubo_methods.py`

- [ ] Extract shared ranking/empirical-percentile helpers without changing
  docking-score direction.
- [ ] Implement BEDROC20 rank-sensitive singleton and pair terms using
  train-only empirical receptor ranks.
- [ ] Implement pair utility and pair synergy coefficients with explicit
  fixed-cardinality semantics.
- [ ] Implement uncertainty/stability coefficient variants only when the
  required seed or fold columns are present.
- [ ] Wrap the existing normalized coverage formulation behind the registry.
- [ ] Implement the auxiliary coverage and rank-bin formulations only for
  their declared input contracts; preserve factorized coefficient metadata.
- [ ] Add formula-level tests with hand-computed small matrices.

## Task 4: Add single-method and comparison configuration paths

**Files:**
- Modify: `src/qubo_receptor_ensemble/full_workflow.py`
- Modify: `src/qubo_receptor_ensemble/experiment.py`
- Modify: `src/qubo_receptor_ensemble/solvers.py`
- Test: `tests/test_full_workflow_stages.py`
- Test: `tests/test_full_workflow_end_to_end.py`

- [ ] Accept a single `problem.methods` item as a compatibility-normalized
  method request, while retaining old `problem.strategy` configs.
- [ ] Accept `problem.mode=compare` with an ordered method list.
- [ ] Reuse one primary matrix and receptor manifest for every compatible
  method.
- [ ] Write method-specific problem, selection, evaluation, and summary JSON
  under `methods/<method_id>/`.
- [ ] Write `method_capabilities.json` and `comparison.json` with BEDROC20 as
  the first-class metric and ROC-AUC as secondary output.
- [ ] Ensure `run --from build_problem` can execute method comparisons without
  re-running preparation, docking, or aggregation.
- [ ] Test output isolation, deterministic ordering, and backward-compatible
  single-method paths.

## Task 5: Update configs, documentation, and operator checks

**Files:**
- Modify: `configs/experiments/stage102a_egfr_full.json`
- Modify: `configs/experiments/stage102a_fa10_full.json`
- Modify: `docs/experiment_workflow_zh.md`
- Modify: `configs/README.md`
- Create: `configs/experiments/stage102a_method_comparison_template.json`
- Test: `tests/test_workflow_catalog.py`

- [ ] Change current Stage102A problem defaults to BEDROC20 explicitly.
- [ ] Add a documented comparison template for compatible methods.
- [ ] Document that trying another method starts at `build_problem` after
  aggregation and does not require new docking.
- [ ] Document capability failures for structural/MD/hardware methods.
- [ ] Add a catalog test that all declared method IDs have metadata and a
  supported execution status.

## Task 6: Full verification

- [ ] Run focused method/config/solver tests.
- [ ] Run the complete Python test suite with the repository source path.
- [ ] Run `python -m compileall -q src scripts`.
- [ ] Validate the updated Stage102A configs without running docking.
- [ ] Run a small comparison fixture through build_problem, solve, and
  evaluate and inspect `comparison.json`.
- [ ] Review the diff and preserve unrelated user changes in the worktree.
