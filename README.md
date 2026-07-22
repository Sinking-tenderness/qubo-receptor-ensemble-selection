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

Stage 5 is in progress on the independent MAPK14/p38alpha target. Eight
label-independent receptor conformers and a 696-ligand development-train panel
have complete three-seed AutoDock Vina 1.2.7 e32 evidence. A preregistered
marginal pair-synergy QUBO passed its train-only gates against matched linear
and nested-greedy comparators, but nested exhaustive search remained stronger.
This is not yet independent validation or quantum advantage.

A fresh scaffold/source-group-disjoint validation panel is frozen with 75
actives, 1,501 decoys, five required receptor columns, and three seeds: 23,640
official CPU Vina jobs. No fresh-validation metric has been calculated and the
test partition remains locked. A Train-160 Uni-Dock GPU pilot was fast but
failed the frozen CPU-equivalence gate, so its scores are not mixed with the
official Vina matrices.

See the [runtime and engine-migration assessment](reports/stage-05/docking_engine_runtime_and_migration_assessment.md)
for measured CPU projections and the evidence that must be rebuilt after an
engine change.

## Milestones

- [x] Validate docking protocols by co-crystal redocking.
- [x] Build reproducible active/decoy score matrices.
- [x] Compare QUBO selection with linear, greedy, single-best, and exhaustive
  development baselines.
- [x] Freeze the MAPK14 marginal pair-synergy candidate on Train-696.
- [ ] Complete the preregistered fresh MAPK14 validation.
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
