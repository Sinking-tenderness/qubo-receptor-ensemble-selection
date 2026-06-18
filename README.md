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

Stage 2 is in progress: molecular docking and virtual-screening foundations.
The first milestone is a reproducible single-receptor baseline with redocking,
active/decoy screening, and ROC-AUC, EF1%, EF5%, and BEDROC evaluation.

## Planned milestones

1. Validate a single-receptor docking protocol by redocking.
2. Build a reproducible active/decoy virtual-screening baseline.
3. Generate a ligand-by-conformer docking-score matrix.
4. Reproduce classical receptor-ensemble selection baselines.
5. Formulate sparse receptor selection as a QUBO.
6. Compare exact, greedy, classical annealing, and quantum-inspired solvers.

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

## Environment

Create the initial Conda environment:

```powershell
conda env create -f environment/environment.yml
conda activate qubo-receptor-ensemble
python -m pip install -e .
python -m pytest
```

Docking engines and structure-preparation tools will be pinned after the
redocking workflow is selected and validated.

## Data policy

Large datasets, prepared ligand libraries, docking poses, molecular-dynamics
trajectories, credentials, and machine-specific files are not committed.
Every experiment should instead record source identifiers, download dates,
software versions, parameters, random seeds, failures, and generated manifests.

## License

MIT License. See [LICENSE](LICENSE).

