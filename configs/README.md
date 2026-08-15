# Configuration Guide

Configuration files are immutable experiment inputs, not competing script
versions. A new file is created when a seed, receptor panel, ligand panel,
engine, search strength, or output boundary changes.

## File Types

- `*.txt`: AutoDock Vina box and search parameters.
- `*_seedN_*.json`: one frozen paired-seed execution.
- `*_aggregation.json`: seed admission and matrix aggregation.
- `*_preregistration.json`: evaluation rules frozen before scores are opened.
- `*_bundle*.json`: deterministic remote package definitions.

## Current MAPK14 Mainline

- `stage05_mk14_fresh_validation_e32_seed0_32vcpu_linux.json`
- `stage05_mk14_fresh_validation_e32_seed1_32vcpu_linux.json`
- `stage05_mk14_fresh_validation_e32_seed2_32vcpu_linux.json`
- `stage05_mk14_fresh_validation_e32_32vcpu_seed_aggregation.json`
- `stage05_mk14_fresh_validation_preregistration.json`
- `stage05_mk14_fresh_validation_execution_amendment01.json`
- `stage05_mk14_fresh_validation_execution_amendment02.json`
- `stage05_mk14_fresh_validation_distributed_execution_amendment03.json`
- `stage05_mk14_fresh_validation_e32_seed1_64vcpu_linux.json`
- `stage05_mk14_fresh_validation_e32_distributed_seed_aggregation.json`
- `stage05_mk14_enopt_xgboost_baseline_preregistration.json`
- `stage05_mk14_enopt_xgboost_fresh_validation_preregistration.json`
- `stage05_mk14_literature_baselines_posthoc.json`
- `stage05_mk14_greedy_failure_screen_posthoc.json`
- `stage07_mk14_unidock113_train160_search_sensitivity.json`
- `stage07b_mk14_unidock113_train160_enhanced_confirmation.json`
- `stage07c_mk14_unidock113_warning_adjudication.json`
- `stage08_mk14_expanded16_structural_selection.json`
- `stage08_mk14_expanded16_structural_selection_audit.json`
- `stage08_mk14_expanded16_unidock_redocking.json`

These files pin official AutoDock Vina 1.2.7, exhaustiveness 32, one output
mode, five receptor columns, and the fresh validation boundary. Amendment 02
selects the available 32-vCPU execution layout of 16 two-CPU Vina processes;
the earlier 64-vCPU files remain immutable audit records. Do not edit frozen
files in place. Any authorized protocol change requires a new configuration
and a dated amendment before fresh-validation scores are generated.

Amendment 03 records the active distributed layout: seed0 and seed2 use their
32-vCPU configurations, while seed1 uses 32 two-CPU processes on a 64-vCPU
instance. The distributed aggregation configuration pins those three exact
config hashes. This changes process placement and concurrency only.

Files containing `unidock` describe a consumed-train experimental branch. Its
tested profiles failed the CPU-equivalence gate and are not current production
configurations.

Files beginning with `stage06_mk14_vinagpu21` describe a separate consumed-
train Vina-GPU 2.1 migration branch. The deterministic execution bridge passed,
but the frozen heuristic-depth engine-equivalence gate remains failed. The only
authorized follow-up is the targeted fixed-depth diagnostic; these files are
not production docking configurations and do not open validation or test data.

The two `enopt_xgboost` files freeze a supplementary classical comparator.
The `literature_baselines_posthoc` file adds fixed LR, sklearn GBT, RF, RFE,
consensus-fusion, and train-only top-three receptor baselines after the fresh
validation result. Its outputs are descriptive and cannot modify the primary
gate or unlock the test split.
The `greedy_failure_screen_posthoc` file compares forward greedy selection
with exact fixed-cardinality enumeration on Train-696 only. It diagnoses local
optima across folds, seeds, budgets, and restricted receptor pools without
reading fresh validation or the locked test.
The `stage07` sensitivity file starts a separate Uni-Dock 1.1.3 development
line. It selects among the official fast, balance, and detail profiles using
consumed Train-160 evidence and never mixes Uni-Dock scores with CPU Vina.
The `stage07b` confirmation file records the preregistered follow-up after no
Stage 07 profile passed. It tests a 512/80 versus 1024/40 factorial extension
and the combined 1024/80 profile, with zero validation or test access.
The `stage07c` file adds one train-only seed and one exact warning replay. It
does not relax rank or BEDROC thresholds; it replaces the blanket warning ban
with a frozen zero-unresolved-warning rule backed by full pose integrity and
exact replay checks.
The `stage08` files extend the existing label-independent max-min sequence to
16 receptors. The first four additions must reproduce the prior expanded-eight
pool exactly. Only the eight new additions are prepared and cognate-redocked;
the Train-696 production matrix remains unauthorized until that gate and its
independent audit pass.
The first binds the completed Train-696 nested-CV fit; the second permits
prediction only after the primary fresh-validation result exists. Neither file
changes the QUBO acceptance criteria or authorizes test release.

The `stage12a` file is a post hoc MAPK14 objective-adequacy diagnostic. It
enumerates all 560 three-receptor subsets from the audited Stage 09 Train-696
matrix, compares the frozen pair-synergy objective with regularized additive
and quadratic surrogates across the existing four folds, and never reads Stage
11 validation or test rows. Any v2 QUBO it emits is development-only evidence
that may be tested only on a new protein target.

## Reuse Rule

For a new target, reuse the general schema and runner but create new target-
specific configurations. Never copy a score path, expected hash, ligand count,
or test-release setting from an older target without regenerating its manifest.
