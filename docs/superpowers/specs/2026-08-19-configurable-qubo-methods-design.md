# Configurable QUBO Methods Design

## Status

Draft for user review. No implementation is included in this document.

## Goal

Make the receptor-selection stage configurable across the historical QUBO
objective, encoding, and solver families used in the repository. A single
experiment must be able to select one method, or reuse the same aggregated
docking matrix to compare several methods. The default ranking objective is
BEDROC with `alpha=20`; ROC-AUC remains an explicitly selectable secondary
diagnostic and is not the default selector objective.

## Current limitations

- The canonical full workflow accepts one `problem.strategy` value.
- The basic QUBO maps `bedroc` to a hard-coded `bedroc_alpha_20` metric.
- The generic QUBO accepts one utility metric and does not expose BEDROC alpha.
- The normalized coverage strategy is imported directly from `scripts`.
- Historical records mix biological objectives, QUBO encodings, solver backends,
  and diagnostic studies. They cannot all be represented as interchangeable
  score-only receptor subset formulas.

## Design

### 1. Method registry

Add a registry keyed by stable `method_id`. Each entry declares:

- the historical provenance and formula family;
- the required input fields and optional artifacts;
- the formulation kind: `qubo`, `qubo_auxiliary`, `qubo_encoded_hubo`, or
  `diagnostic`;
- supported solver backends;
- default parameters and whether BEDROC alpha is applicable;
- a builder that returns the existing solver-independent `Problem` contract.

The registry is the only dispatch point for method selection. Historical
scripts may remain as provenance references, but new workflow execution must
not import an arbitrary script by path.

### 2. Configuration compatibility

Keep the current single-method form valid:

```json
"problem": {
  "type": "receptor_subset",
  "strategy": "qubo",
  "target_size": 3,
  "utility_metric": "bedroc",
  "bedroc_alpha": 20.0,
  "weights": {
    "redundancy": 0.25,
    "count": 0.1,
    "size": 10.0
  }
}
```

For method comparison, add an explicit list while retaining the same input
matrix and receptor manifest:

```json
"problem": {
  "type": "receptor_subset",
  "mode": "compare",
  "methods": [
    {
      "id": "basic_utility",
      "target_size": 3,
      "utility_metric": "bedroc",
      "bedroc_alpha": 20.0
    },
    {
      "id": "bedroc20_pair_synergy",
      "target_size": 3,
      "bedroc_alpha": 20.0
    }
  ]
}
```

An omitted utility metric inherits `bedroc`; an omitted alpha inherits
`20.0`. A method may explicitly request `roc_auc`, but this must be visible in
the method snapshot and comparison output.

### 3. Historical method layers

The implementation will distinguish three layers instead of treating every
historical stage as a separate objective:

#### Objective families

- basic singleton utility plus redundancy;
- pair utility and pair synergy;
- BEDROC20 rank-sensitive pair complementarity;
- uncertainty-shrunk or stability-weighted coefficients;
- active coverage and decoy-exposure objectives;
- normalized and multiscale coverage objectives;
- quality-plateau and functional-diversity portfolio objectives;
- rank-bin approximations of the continuous BEDROC objective;
- signed higher-order Mobius/HUBO objectives.

#### Encodings

- ordinary fixed-cardinality penalty QUBO;
- exact fixed-cardinality coefficient form;
- auxiliary-variable coverage QUBO;
- rank-bin auxiliary-variable QUBO;
- Rosenberg quadratization for retained cubic terms;
- constraint-aware and quality-shell encodings.

#### Solvers and diagnostics

- exact enumeration;
- greedy and local-search comparators;
- classical sampler or CQM adapters where the input contract exists;
- Dirac or hardware-specific encodings only behind an explicit backend;
- objective-alignment, precision, noise, and sampler-stability studies as
  diagnostics, not selectable biological objectives.

### 4. Capability validation

`validate` must inspect method requirements before any stage runs. A method
that needs structural states, MD frame metadata, group assignments, auxiliary
coverage inputs, or a hardware backend must fail with a specific missing-input
message when those artifacts are absent. It must never silently fall back to a
score-only formula.

Methods that are registered for historical completeness but cannot run for the
current Stage102A data will be reported as `unsupported_for_input` in a method
capability report. Score-matrix methods remain directly runnable on the
current `primary_matrix.csv`.

### 5. Output layout

Single-method runs retain the existing paths and JSON shape. Comparison runs
write method-specific artifacts under the run directory:

```text
methods/<method_id>/problem.json
methods/<method_id>/selection.json
methods/<method_id>/evaluation.json
methods/<method_id>/summary.json
method_capabilities.json
comparison.json
```

Every method result records its method ID, provenance, formula identifier,
parameters, BEDROC alpha when applicable, required inputs, solver backend,
objective convention, and selection split. The comparison file reports
BEDROC20 first and retains ROC-AUC as a secondary column.

### 6. Leakage and reproducibility rules

- Method construction and selection use only the configured training rows.
- The locked test split remains unavailable to selectors.
- All methods in one comparison use the same matrix, receptor pool, split, and
  target-size policy unless their configuration explicitly declares otherwise.
- The resolved method configuration is copied into the run snapshot.
- Existing docking and preparation artifacts are reusable; trying another
  method begins at `build_problem`.

## Non-goals

- Recreating missing historical datasets or fabricating structural features.
- Treating every historical diagnostic script as a production selector.
- Changing raw-data preparation, receptor alignment, box calculation, or
  docking behavior.
- Removing ROC-AUC from reports; it remains a secondary diagnostic.

## Acceptance criteria

1. Existing single-method configurations continue to validate and run.
2. Generic methods default to BEDROC20 and expose `bedroc_alpha` in their
   serialized formulation.
3. A historical BEDROC20 pair-synergy method can be selected from config.
4. Multiple compatible methods can be run from one aggregated matrix without
   repeating docking.
5. Method capability failures identify missing inputs or unsupported backends.
6. Results are auditable and comparable with BEDROC20 as the primary metric.
7. Unit tests cover defaults, alpha propagation, method dispatch, capability
   validation, backward compatibility, and multi-method output isolation.

## Review question

This design assumes that historically specialized methods are registered and
validated, but only methods whose required inputs exist are executed in a
given run. It does not silently reinterpret a structural or hardware method as
a score-only method. Confirm this boundary before implementation.
