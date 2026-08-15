from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


FILES = (
    "configs/stage92_bace1_group_robust_hardness_adjudication.json",
    "configs/stage93_bace1_seed_consensus_hardness_diagnostic.json",
    "configs/stage94_bace1_series_assignment_facility_location.json",
    "data/stage92a_bace1_target_id_metadata_adjudication_result.json",
    "data/stage93_bace1_seed_consensus_hardness_diagnostic_result.json",
    "data/stage94_bace1_series_assignment_facility_location_result.json",
    "data/stage94_bace1_series_assignment_facility_location_audit.json",
    "results/runs/stage92a_bace1_target_id_metadata_adjudication/scores_target_id_amended.csv",
    "results/runs/stage93_bace1_seed_consensus_hardness_diagnostic/classical_baselines.csv",
    "results/runs/stage94_bace1_series_assignment_facility_location/classical_baselines.csv",
    "results/runs/stage94_bace1_series_assignment_facility_location/exact_assignments.csv",
    "reports/stage-93/bace1_seed_consensus_hardness_diagnostic.md",
    "reports/stage-94/bace1_series_assignment_facility_location.md",
    "scripts/adjudicate_stage91c_bace1_target_id_metadata.py",
    "scripts/run_stage93_bace1_seed_consensus_hardness_diagnostic.py",
    "scripts/run_stage94_bace1_series_assignment_facility_location.py",
    "scripts/audit_stage94_bace1_series_assignment_facility_location.py",
    "tests/test_stage92_bace1_group_robust_hardness.py",
    "tests/test_stage94_bace1_series_assignment_facility_location.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "deliverables/stage94_bace1_series_assignment_facility_location_core_v1.tar.gz"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, list(FILES))
    result.update(
        {
            "operation": "Stage92a metadata adjudication plus Stage93 and Stage94 BACE1 hardness diagnostics",
            "stage93_hardness_gate_passed": False,
            "stage94_hardness_gate_passed": False,
            "new_docking_jobs": 0,
            "quantum_jobs": 0,
            "same_data_objective_engineering_closed": True,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
