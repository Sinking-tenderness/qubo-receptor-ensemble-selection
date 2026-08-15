"""Build a deterministic Stage58c-59 PPARD functional-gate evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


PATHS = (
    "configs/stage55_ppard_small_pilot_preregistration.json",
    "configs/stage58c_ppard_target_id_amendment.json",
    "configs/stage59_ppard_functional_complementarity_gate.json",
    "data/stage54_future_target_intake_criteria.json",
    "data/stage58b_ppard_pilot96_unidock113_production_audit.json",
    "data/stage58c_ppard_target_id_amendment_result.json",
    "data/stage58c_ppard_target_id_amendment_audit.json",
    "data/stage59_ppard_functional_complementarity_gate_result.json",
    "data/stage59_ppard_functional_complementarity_gate_audit.json",
    "data/processed/stage58a_ppard_pilot96_unidock_pdbqt_manifest.csv",
    "data/processed/stage58b_ppard_stage57_passing29_receptor_manifest.csv",
    "results/runs/stage58b_ppard_pilot96_unidock113_production/summary.json",
    "results/runs/stage58b_ppard_pilot96_unidock113_production/scores.csv",
    "results/runs/stage58c_ppard_target_id_amendment/scores.csv",
    "results/runs/stage59_ppard_functional_complementarity_gate/receptor_metrics.csv",
    "results/runs/stage59_ppard_functional_complementarity_gate/pair_metrics.csv",
    "results/runs/stage59_ppard_functional_complementarity_gate/fold_gate.csv",
    "reports/stage-59/ppard_functional_complementarity_gate.md",
    "scripts/amend_stage58c_ppard_target_id.py",
    "scripts/audit_stage58c_ppard_target_id.py",
    "scripts/run_stage59_ppard_functional_complementarity_gate.py",
    "scripts/audit_stage59_ppard_functional_complementarity_gate.py",
    "tests/test_stage58c_59_ppard_functional_gate.py",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="ascii"))


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    if any(not (root / path).is_file() for path in PATHS):
        raise ValueError("Stage59 evidence bundle is incomplete")
    amendment = read_json(root / "data/stage58c_ppard_target_id_amendment_audit.json")
    result = read_json(root / "data/stage59_ppard_functional_complementarity_gate_result.json")
    audit = read_json(root / "data/stage59_ppard_functional_complementarity_gate_audit.json")
    if amendment.get("status") != "stage58c_ppard_target_id_amendment_independent_audit_ok":
        raise ValueError("Stage58c independent audit did not pass")
    if result.get("status") != "stage59_ppard_functional_complementarity_gate_complete":
        raise ValueError("Stage59 result did not complete")
    if audit.get("status") != "stage59_ppard_functional_complementarity_independent_audit_ok":
        raise ValueError("Stage59 independent audit did not pass")
    if not result["decision"]["functional_complementarity_gate_passed"]:
        raise ValueError("Stage59 functional-complementarity gate did not pass")
    if any(
        int(result["data_boundary"][key]) != 0
        for key in (
            "fresh_validation_rows_read", "locked_test_rows_read",
            "new_docking_jobs", "quantum_hardware_jobs",
        )
    ):
        raise ValueError("Stage59 crossed a protected boundary")
    return sorted(PATHS)


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
            "operation": "Stage58c-59 PPARD preregistered functional-complementarity gate evidence",
            "target_id": "PPARD",
            "functional_complementarity_gate_passed": True,
            "receptor_count": 29,
            "pilot_ligand_count": 96,
            "single_receptor_count": 29,
            "pair_count": 406,
            "outer_fold_count": 4,
            "fresh_validation_rows": 0,
            "locked_test_rows": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
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
