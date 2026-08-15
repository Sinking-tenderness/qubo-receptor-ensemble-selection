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
    "configs/stage91_bace1_group_robust_rescue_preregistration.json",
    "configs/stage91c_bace1_group_robust_development_docking_preregistration.json",
    "configs/stage92_bace1_group_robust_hardness_adjudication.json",
    "data/stage91_bace1_group_robust_rescue_preregistration_result.json",
    "data/stage92_bace1_group_robust_hardness_adjudication_result.json",
    "data/stage92_bace1_group_robust_hardness_adjudication_audit.json",
    "analysis/stage91c_bace1_result_20260812/results/runs/stage91c_bace1_chembl365_unidock113_production/summary.json",
    "analysis/stage91c_bace1_result_20260812/data/stage91c_bace1_chembl365_unidock113_production_audit.json",
    "analysis/stage91c_bace1_result_20260812/results/runs/stage91c_bace1_chembl365_unidock113_production/primary_median_score_matrix.csv",
    "analysis/stage91c_bace1_result_20260812/results/runs/stage91c_bace1_chembl365_unidock113_production/sensitivity_minimum_score_matrix.csv",
    "analysis/stage91c_bace1_result_20260812/data/processed/stage91b_bace1_chembl365_unidock_pdbqt_manifest.csv",
    "results/runs/stage92_bace1_group_robust_hardness_adjudication/classical_baselines.csv",
    "results/runs/stage92_bace1_group_robust_hardness_adjudication/selected_receptors.csv",
    "reports/stage-92/bace1_group_robust_hardness_adjudication.md",
    "scripts/run_stage92_bace1_group_robust_hardness_adjudication.py",
    "scripts/audit_stage92_bace1_group_robust_hardness_adjudication.py",
    "tests/test_stage92_bace1_group_robust_hardness.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "deliverables/stage92_bace1_group_robust_hardness_adjudication_core_v1.tar.gz"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, list(FILES))
    result.update(
        {
            "operation": "Stage91c matrix acceptance and Stage92 frozen hardness adjudication",
            "source_pair_count": 37230,
            "state_count_reenumerated": 1344904,
            "hardness_gate_passed": False,
            "confirmation_a_authorized": False,
            "quantum_hardware_authorized": False,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
