"""Build the Stage58a PPARD Pilot-96 ligand-preparation input bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle
from scripts.experimental.unidock.prepare_development_ligand_inputs import read_json
from scripts.experimental.unidock.prepare_stage58a_ppard_pilot96_inputs import validate_source


CONFIG = "configs/stage58a_ppard_pilot96_unidock_input_preparation.json"
PATHS = (
    "configs/stage55_ppard_small_pilot_preregistration.json",
    "configs/stage56_ppard_ligand_panel_allocation.json",
    CONFIG,
    "data/stage56_ppard_ligand_panel_allocation_summary.json",
    "data/processed/stage56_ppard_pilot96_ligand_manifest.csv",
    "data/processed/stage56_ppard_pilot96_fold_assignments.csv",
    "data/stage57_ppard_cognate_redocking_summary.json",
    "data/processed/stage57_ppard_receptor_gate_results.csv",
    "environment/stage08_unidock_gpu.yml",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/prepare_ligand_3d_sdf.py",
    "scripts/batch_prepare_ligand_pdbqt.py",
    "scripts/audit_stage58a_ppard_pilot96_inputs.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/prepare_development_ligand_inputs.py",
    "scripts/experimental/unidock/prepare_stage58a_ppard_pilot96_inputs.py",
    "scripts/experimental/unidock/run_stage58a_ppard_pilot96_input_preparation_remote.sh",
    "scripts/experimental/unidock/build_stage58a_ppard_pilot96_input_bundle.py",
    "reports/stage-58/ppard_pilot96_input_preparation.md",
    "tests/test_stage58a_ppard_pilot96_input_preparation.py",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = read_json(root / CONFIG)
    rows, _, labels = validate_source(root, config)
    if len(rows) != 96 or labels != {"active": 48, "decoy": 48}:
        raise ValueError("Stage58a frozen Pilot-96 differs")
    fold_labels = {
        (row["pilot_outer_fold"], row["label"]) for row in rows
    }
    if fold_labels != {
        (str(fold), label) for fold in range(4) for label in ("active", "decoy")
    }:
        raise ValueError("Stage58a frozen fold labels differ")
    stage57 = read_json(root / "data/stage57_ppard_cognate_redocking_summary.json")
    if (
        stage57["status"] != "stage57_ppard_cognate_redocking_gate_ok"
        or int(stage57["passed_receptor_count"]) != 29
        or not bool(stage57["technical_gate_pass"])
    ):
        raise ValueError("Stage58a is not authorized by Stage57")
    if any(
        marker in path.lower()
        for path in PATHS
        for marker in ("fresh_validation", "locked_test", "protected/")
    ):
        raise ValueError("Stage58a bundle crossed a protected split boundary")
    return sorted(set(PATHS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, bundle_paths(root))
    result.update(
        {
            "operation": "Stage58a PPARD outcome-blind Pilot-96 checkpointed ligand input preparation",
            "target_id": "PPARD",
            "analysis_class": "prospective outcome-blind development pilot",
            "development_active_count": 48,
            "development_decoy_count": 48,
            "development_ligand_count": 96,
            "frozen_receptor_count": 29,
            "future_seed_count": 3,
            "future_pair_count": 8352,
            "gpu_docking_jobs_in_this_bundle": 0,
            "fresh_validation_structures_prepared": 0,
            "locked_test_structures_prepared": 0,
            "resume_supported": True,
        }
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
