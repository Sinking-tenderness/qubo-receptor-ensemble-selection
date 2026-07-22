"""Discover and launch the small supported workflow surface.

The repository keeps historical experiment entry points for reproducibility.
This catalog is the supported starting point for new work.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Workflow:
    path: str
    category: str
    status: str
    description: str
    runnable: bool = True


WORKFLOWS: dict[str, Workflow] = {
    "audit-smiles": Workflow(
        "scripts/check_ligand_smiles.py",
        "core",
        "supported",
        "Audit ligand SMILES before structure preparation.",
    ),
    "prepare-ligand-3d": Workflow(
        "scripts/prepare_ligand_3d_sdf.py",
        "core",
        "supported",
        "Generate explicit-hydrogen 3D SDF ligand structures.",
    ),
    "prepare-ligand-pdbqt": Workflow(
        "scripts/batch_prepare_ligand_pdbqt_parallel.py",
        "core",
        "supported",
        "Prepare ligand PDBQT files with resumable parallel workers.",
    ),
    "align-receptor": Workflow(
        "scripts/align_receptor_structure.py",
        "core",
        "supported",
        "Align a receptor to a shared coordinate frame.",
    ),
    "prepare-receptor": Workflow(
        "scripts/prepare_receptor.py",
        "core",
        "supported",
        "Prepare and audit a receptor PDBQT file.",
    ),
    "redocking-rmsd": Workflow(
        "scripts/evaluate_redocking_rmsd.py",
        "core",
        "supported",
        "Calculate symmetry-aware redocking RMSD.",
    ),
    "dock-vina": Workflow(
        "scripts/batch_vina_docking_parallel.py",
        "core",
        "supported",
        "Run resumable parallel AutoDock Vina jobs.",
    ),
    "build-score-matrix": Workflow(
        "scripts/build_score_matrix.py",
        "core",
        "supported",
        "Build representative long tables and ligand-by-receptor matrices.",
    ),
    "aggregate-seeds": Workflow(
        "scripts/aggregate_seed_replicates.py",
        "core",
        "supported",
        "Audit and aggregate the current paired-seed matrix schema.",
    ),
    "evaluate-screening": Workflow(
        "scripts/evaluate_virtual_screening.py",
        "core",
        "supported",
        "Calculate ranking and virtual-screening metrics.",
    ),
    "fit-ensemble": Workflow(
        "scripts/run_development_scaffold_cv_gate.py",
        "selection",
        "supported",
        "Fit receptor-selection methods inside development-only scaffold CV.",
    ),
    "fit-enopt-xgboost": Workflow(
        "scripts/fit_enopt_xgboost_baseline.py",
        "selection",
        "frozen",
        "Frozen Train-696 EnOpt-style nonlinear comparator; do not retune.",
        runnable=False,
    ),
    "md-build": Workflow(
        "scripts/build_openmm_system.py",
        "molecular-dynamics",
        "supported",
        "Build an audited solvated OpenMM system.",
    ),
    "md-equilibrate": Workflow(
        "scripts/run_openmm_equilibration.py",
        "molecular-dynamics",
        "supported",
        "Run resumable minimization, NVT, and NPT equilibration.",
    ),
    "md-production": Workflow(
        "scripts/run_openmm_production.py",
        "molecular-dynamics",
        "supported",
        "Run resumable NPT production dynamics.",
    ),
    "md-analyze": Workflow(
        "scripts/analyze_md_trajectory.py",
        "molecular-dynamics",
        "supported",
        "Analyze aligned backbone and pocket trajectory stability.",
    ),
    "fresh-validation-build": Workflow(
        "scripts/build_stage05_mk14_fresh_validation_remote_bundle.py",
        "current-mapk14",
        "frozen",
        "Build the preregistered CPU Vina fresh-validation bundle.",
    ),
    "fresh-validation-evaluate": Workflow(
        "scripts/evaluate_stage05_mk14_fresh_validation.py",
        "current-mapk14",
        "restricted",
        "One-time preregistered evaluation after returned scores are admitted.",
        runnable=False,
    ),
    "fresh-validation-xgboost": Workflow(
        "scripts/evaluate_enopt_xgboost_fresh_validation.py",
        "current-mapk14",
        "restricted",
        "Supplementary frozen XGBoost evaluation after the primary gate.",
        runnable=False,
    ),
    "unidock-equivalence": Workflow(
        "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
        "experimental",
        "failed-gate",
        "Consumed-train GPU equivalence diagnostics; not a CPU Vina replacement.",
    ),
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list supported workflow aliases")

    show = subparsers.add_parser("show", help="describe one workflow alias")
    show.add_argument("name", choices=sorted(WORKFLOWS))

    run = subparsers.add_parser("run", help="run a supported workflow alias")
    run.add_argument("name", choices=sorted(WORKFLOWS))
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def print_workflow(name: str, workflow: Workflow) -> None:
    print(f"{name}")
    print(f"  category: {workflow.category}")
    print(f"  status: {workflow.status}")
    print(f"  path: {workflow.path}")
    print(f"  purpose: {workflow.description}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        for category in sorted({item.category for item in WORKFLOWS.values()}):
            print(f"[{category}]")
            for name, workflow in sorted(WORKFLOWS.items()):
                if workflow.category == category:
                    print(f"  {name:<28} {workflow.status:<12} {workflow.description}")
        return 0

    workflow = WORKFLOWS[args.name]
    if args.command == "show":
        print_workflow(args.name, workflow)
        return 0

    if not workflow.runnable:
        raise SystemExit(
            f"{args.name} is restricted. Review its preregistration and run the "
            "pinned script directly only after score admission."
        )
    script = repository_root() / workflow.path
    if not script.is_file():
        raise FileNotFoundError(script)
    forwarded = list(args.arguments)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    completed = subprocess.run([sys.executable, str(script), *forwarded], check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
