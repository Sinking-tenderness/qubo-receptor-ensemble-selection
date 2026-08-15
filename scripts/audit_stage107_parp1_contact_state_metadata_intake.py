"""Independently audit Stage107's public metadata gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def audit(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    output = config["outputs"]
    result = json.loads((root / output["summary_json"]).read_text(encoding="utf-8"))
    if result["status"] != "stage107_parp1_contact_state_metadata_intake_no_go":
        raise ValueError("unexpected Stage107 status")
    minimum = int(config["discovery_rules"]["minimum_metadata_eligible_count"])
    if int(result["counts"]["metadata_eligible_count"]) >= minimum:
        raise ValueError("structural-count sub-gate unexpectedly passed")
    if bool(result["counts"]["reference_eligible"]):
        raise ValueError("3L3M should fail the frozen reference rule")
    if bool(result["counts"]["gate_passes"]):
        raise ValueError("legacy gate should be No-Go")
    if len(result["eligible_pdb_ids"]) != int(result["counts"]["metadata_eligible_count"]):
        raise ValueError("eligible PDB count mismatch")
    if any(value != 0 for value in result["data_boundary"].values()):
        raise ValueError("Stage107 crossed the data boundary")
    decision = result["decision"]
    if decision["coordinate_structural_audit_authorized"]:
        raise ValueError("coordinate audit must remain locked after No-Go")
    if any(decision[key] for key in ("ligand_preparation_authorized", "redocking_authorized", "production_docking_authorized", "parp1_fresh_validation_released", "parp1_locked_test_released", "quantum_hardware_authorized")):
        raise ValueError("Stage107 improperly released a later task")
    return {
        "schema_version": "1.0",
        "status": "stage107_independent_audit_ok",
        "metadata_eligible_count": result["counts"]["metadata_eligible_count"],
        "structural_count_passes": False,
        "reference_eligible": False,
        "coordinate_structural_audit_authorized": False,
        "later_stages_locked": True,
        "data_boundary": result["data_boundary"]
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage107_parp1_contact_state_metadata_intake.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    record = audit(root, args.config)
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    path = root / config["outputs"]["audit_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
