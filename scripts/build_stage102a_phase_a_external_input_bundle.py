from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


STATIC_FILES = [
    "configs/stage102_prospective_marginal_learning.json",
    "data/stage102_prospective_marginal_learning_readiness.json",
    "data/stage13g_egfr_cognate_redocking_failure_adjudication.json",
    "data/stage14e_fa10_cognate_redocking_failure_adjudication.json",
    "data/processed/stage13e_egfr_prepared_receptor_manifest.csv",
    "data/processed/stage14c_fa10_prepared_receptor_manifest.csv",
    "data/processed/stage102a_egfr_passing_receptor_manifest.csv",
    "data/processed/stage102a_fa10_passing_receptor_manifest.csv",
    "data/stage102a_phase_a_receptor_freeze_summary.json",
    "data/raw/external_targets/egfr_dude/egfr/actives_final.ism",
    "data/raw/external_targets/egfr_dude/egfr/decoys_final.ism",
    "data/raw/external_targets/fa10_dude/fa10/actives_final.ism",
    "data/raw/external_targets/fa10_dude/fa10/decoys_final.ism",
    "environment/stage08_unidock_gpu.yml",
    "reports/stage-102/prospective_marginal_learning_plan.md",
    "scripts/__init__.py",
    "scripts/experimental/__init__.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/freeze_stage102a_phase_a_receptors.py",
    "scripts/allocate_stage102a_phase_a_ligands.py",
    "scripts/prepare_stage102a_phase_a_inputs.py",
    "scripts/prepare_ligand_3d_sdf.py",
    "scripts/batch_prepare_ligand_pdbqt.py",
    "scripts/evaluate_redocking_rmsd.py",
    "scripts/experimental/unidock/prepare_development_ligand_inputs.py",
    "scripts/experimental/unidock/run_stage102a_phase_a_production.py",
    "scripts/experimental/unidock/run_stage09_mk14_train696_production.py",
    "scripts/experimental/unidock/run_unidock_batch_targeted.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
    "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
    "scripts/experimental/unidock/run_stage102a_phase_a_remote.sh",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files = list(STATIC_FILES)
    for target in ("egfr", "fa10"):
        manifest = root / f"data/processed/stage102a_{target}_passing_receptor_manifest.csv"
        for row in read_csv(manifest):
            files.extend([row["receptor_pdbqt"], row["receptor_preparation_summary"]])
    files = list(dict.fromkeys(files))
    result = write_bundle(root, args.output, files)
    result.update({
        "operation": "Stage102A EGFR and FA10 Phase A external GPU input",
        "targets": ["EGFR", "FA10"],
        "receptor_count": 25,
        "ligand_count_to_prepare": 1200,
        "receptor_ligand_seed_pair_count": 45000,
        "required_gpu": "one NVIDIA RTX 4090-class GPU",
        "existing_environment_reusable": "qubo-unidock-stage08",
        "phase_b_parp1_released": False,
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
