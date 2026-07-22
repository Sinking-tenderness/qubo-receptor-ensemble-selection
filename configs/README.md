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

The two `enopt_xgboost` files freeze a supplementary classical comparator.
The first binds the completed Train-696 nested-CV fit; the second permits
prediction only after the primary fresh-validation result exists. Neither file
changes the QUBO acceptance criteria or authorizes test release.

## Reuse Rule

For a new target, reuse the general schema and runner but create new target-
specific configurations. Never copy a score path, expected hash, ligand count,
or test-release setting from an older target without regenerating its manifest.
