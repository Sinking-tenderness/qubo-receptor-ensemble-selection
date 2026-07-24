# QUBO Receptor Ensemble Selection

Research code for QUBO-guided sparse receptor conformational ensemble
selection from large AlphaFold/MD-derived conformer pools, with the goal of
improving early enrichment in flexible virtual screening under a limited
docking budget.

## Research question

Can a QUBO formulation select a small, physically diverse receptor ensemble
that matches or improves virtual-screening early enrichment compared with
classical receptor-selection baselines?

This project optimizes **receptor conformer subsets**. It does not claim to
introduce ensemble docking, molecular docking, or ligand-pose QUBO methods.

## Current status

Stage 4 CDK2 development and its preregistered locked-test evaluation are
complete. The fixed single-receptor protocol achieved ROC-AUC 0.806 on the
independent 40-ligand CDK2 test; that test set is now permanently consumed and
must not be reused for fitting.

Stage 5 MAPK14/p38alpha development and preregistered fresh validation are
complete. The validation used 75 actives, 1,501 decoys, five receptor columns,
three seeds, and 23,640 successful official AutoDock Vina 1.2.7 jobs. The
pair-synergy QUBO passed the frozen fresh-validation checks against matched
linear, exhaustive, and single-receptor comparators. However, QUBO and nested
greedy selected the same three receptors and therefore made identical
predictions. This supports the receptor subset, but it does not demonstrate a
QUBO-over-greedy or quantum advantage. The test partition remains locked.

A Train-160 Uni-Dock GPU pilot was fast but failed the frozen CPU-equivalence
gate, so its scores are not mixed with official Vina matrices. The Stage 6
AutoDock Vina-GPU 2.1 single-pair pilot also failed its complete frozen gate:
aggregate scores were close, but two receptor-seed rank groups and the 5x speed
threshold failed. Its hash-pinned deterministic-batch bridge subsequently
reproduced all 2,400 GPU scores and pose hashes exactly and reached 7.536x
recorded 32-vCPU throughput. Only a bounded fixed-search-depth diagnostic on
the two failed consumed-train groups is now authorized.

See the [runtime and engine-migration assessment](reports/stage-05/docking_engine_runtime_and_migration_assessment.md)
for measured CPU projections and the evidence that must be rebuilt after an
engine change.

## Milestones

- [x] Validate docking protocols by co-crystal redocking.
- [x] Build reproducible active/decoy score matrices.
- [x] Compare QUBO selection with linear, greedy, single-best, and exhaustive
  development baselines.
- [x] Freeze the MAPK14 marginal pair-synergy candidate on Train-696.
- [x] Complete the preregistered fresh MAPK14 validation.
- [x] Complete the isolated Vina-GPU 2.1 Train-160 equivalence pilot and
  deterministic-batch execution bridge.
- [ ] Complete the bounded Vina-GPU fixed-search-depth diagnostic and, only if
  it selects a candidate, a full uniform-depth Train-160 confirmation.
- [ ] Decide whether a separately preregistered locked-test release is
  justified; no automatic release is allowed.

## Repository layout

```text
configs/       Versioned experiment configurations
data/          Data manifests and documentation; large datasets stay local
environment/   Reproducible software environments
ligands/       Ligand preparation notes and small examples
notebooks/     Exploratory analysis only
receptors/     Receptor preparation notes and small examples
reports/       Stage reports and research notes
results/       Small summary tables and publication-ready figures
scripts/       Command-line workflow entry points
src/           Reusable Python package code
tests/         Automated tests
```

The repository retains experiment-specific scripts for hash-pinned
reproducibility. New users should start from the supported catalog instead of
guessing an execution order from filenames:

```powershell
python .\scripts\workflow.py list
python .\scripts\workflow.py show dock-vina
```

See [the script guide](scripts/README.md) for the canonical pipeline and the
boundary between supported, frozen, audit, and experimental commands.

## Environment

Create the initial Conda environment:

```powershell
conda env create -f environment/environment.yml
conda activate qubo-receptor-ensemble
python -m pip install -e .
python -m pytest
```

Experiment configurations pin the docking engine, preparation tools, inputs,
parameters, seeds, and expected hashes. Do not substitute an engine inside an
existing score matrix or validation protocol.

## Data policy

Large datasets, prepared ligand libraries, docking poses, molecular-dynamics
trajectories, credentials, and machine-specific files are not committed.
Every experiment should instead record source identifiers, download dates,
software versions, parameters, random seeds, failures, and generated manifests.

## Incremental workflow

Each completed teaching or research module should end with a focused commit:

1. run the smallest relevant validation or test;
2. update documentation and experiment manifests;
3. review `git status` and the intended diff;
4. commit only the files belonging to that module;
5. push the commit to GitHub.

See [the Stage 2 teaching prompt](reports/stage-02/STAGE_02_TEACHING_PROMPT.md)
for the complete teaching plan with GitHub checkpoints at coherent milestones.

## License

MIT License. See [LICENSE](LICENSE).
