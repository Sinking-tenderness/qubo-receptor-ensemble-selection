# Adaptive Cardinality (k) Implementation Plan

> **For the implementation agent:** use the repository's existing `k_selection` abstraction and execute this plan incrementally with tests first.

**Goal:** Add an opt-in schema 3.0 policy that tests receptor cardinality sequentially from `k=1` and selects the smallest `k` whose inner-fold, scaffold-bootstrap evidence supports a positive functional gain. Existing fixed `problem.target_size` configurations remain unchanged.

**Recommended design:** Treat adaptive cardinality as a controller around the existing QUBO builder and solver. For every `k-1 -> k` transition, solve candidate subsets on inner training folds, evaluate out-of-inner-fold BEDROC20 and active-minus-decoy rescue, bootstrap scaffold groups, and stop permanently at the first failed transition. The final selected `k` is then used for one normal full-development QUBO solve.

**Safety boundary:** The first version implements only the mechanistic bootstrap-LCB rule. It does not train a cross-target Ridge model, read outer/test labels for selection, or claim that the rejected Stage102B development candidates passed a scientific gate.

**Files:**
- Create `src/qubo_receptor_ensemble/adaptive_cardinality.py` for pure data validation, inner-fold scoring, scaffold bootstrap, rescue contrasts, and the auditable decision object.
- Modify `src/qubo_receptor_ensemble/k_selection.py` only if the new policy can be exposed without coupling it to full-workflow I/O; preserve the existing `best_metric` policy.
- Modify `src/qubo_receptor_ensemble/full_workflow.py` to validate an optional `problem.k_policy` block and its candidate bounds.
- Modify `src/qubo_receptor_ensemble/experiment.py` so schema 3.0 `build_problem` computes and persists an adaptive decision before building the final problem, then injects the selected `target_size` into the final problem configuration.
- Modify `src/qubo_receptor_ensemble/solvers.py` only if an existing solver contract needs a narrow compatibility adjustment; do not alter QUBO objective semantics.
- Add `tests/test_adaptive_cardinality.py` for the policy and `tests/test_full_workflow_stages.py` or a focused new test for schema 3.0 integration.
- Update `docs/experiment_workflow_zh.md` and `configs/experiments/README.md` with the opt-in configuration and the evidence boundary after implementation is verified.

### Task 1: Define the policy contract

- Input rows must include `ligand_id`, `label`, scaffold/group identity, an outer-fold identifier, and one score column per receptor. Lower docking scores are better.
- Candidate cardinalities are positive integers, strictly increasing, and bounded by the receptor pool; default candidates are `(1, 2, 3)`.
- The policy starts at `k=1`. Each next transition is accepted only when both conditions hold: scaffold-bootstrap lower confidence bound for paired BEDROC20 gain is strictly positive, and the mean of top-1% and top-5% active-minus-decoy rescue contrasts is strictly positive.
- The first failed transition ends the scan. Ties and non-finite values fail closed to the smaller `k`.
- The decision must expose selected `k`, `need_multi_conformation`, per-transition diagnostics, bootstrap settings, and `uses_outer_labels: false`.

### Task 2: Write failing tests first

- Test that a positive paired gain with positive rescue accepts `1 -> 2` and returns `selected_k=2`.
- Test that a non-positive bootstrap lower bound stops at `k=1` and never evaluates `k=3`.
- Test that a negative rescue contrast stops even when the mean BEDROC gain is positive.
- Test deterministic bootstrap results for a fixed seed and rejection of missing scaffold/fold/score fields.
- Test schema 3.0 accepts an opt-in policy and fixed configurations remain unchanged.
- Test full-workflow problem construction persists the decision and uses its `selected_k` as `target_size`.

### Task 3: Implement the pure policy

- Normalize rows and validate labels, folds, scaffold groups, receptor scores, and candidate cardinalities.
- For each outer fold, use only its training rows and split those rows into deterministic inner folds by scaffold group; compute candidate QUBO selections from inner-training rows and score them on the held inner fold.
- Aggregate out-of-inner-fold ranks for `k-1` and `k`, compute paired BEDROC20 gains, and compute rescue contrasts at top-1% and top-5%.
- Resample scaffold groups with replacement for the configured number of iterations and calculate the configured lower quantile.
- Aggregate transition diagnostics across outer folds without using their labels for selection; select the smallest `k` whose transition passes.

### Task 4: Integrate schema 3.0

- Add an optional `problem.k_policy` validation branch. The default remains fixed `target_size`.
- At `build_problem`, merge metadata from `prepared_ligand_manifest` into matrix rows, run the policy when configured, write `adaptive_cardinality.json`, and set the final problem request's `target_size` to `selected_k`.
- Preserve comparison mode as fixed-cardinality in this first version; reject combining adaptive policy with multi-method comparison until a separate contract exists.
- Include the decision in `problem.json`, `selection.json`, `summary.json`, and manifest records.
- Ensure `solve` still calls the existing `build_problem` and `solve_problem` contracts.

### Task 5: Verify and document

- Run the focused adaptive and full-workflow tests first, then the relevant existing k-selection and solver tests.
- Run `git diff --check` and inspect the final diff for unintended changes.
- Run the full test collection where the environment permits; report the existing RDKit/XGBoost dependency limitation if it remains.
- Document that adaptive selection is experimental, opt-in, and not a Phase-A scientific validation result.
