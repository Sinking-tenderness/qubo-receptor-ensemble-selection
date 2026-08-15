"""Independently audit the Stage110 THRB amendment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(root: Path, config_path: Path) -> dict:
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    result = json.loads((root / config["outputs"]["summary_json"]).read_text(encoding="utf-8"))
    if result["status"] != "stage112_historical_candidate_pool_amendment01_ok":
        raise ValueError("amendment did not complete")
    historical = result["historical_record"]
    corrected = result["amended_thrb_entry"]
    registry = result["corrected_registry_state"]
    if not historical["stage110_file_preserved"] or historical["superseded_for_target_ids"] != ["THRB"]:
        raise ValueError("historical preservation differs")
    if corrected["historical_metadata_eligible_count"] != 19 or corrected["corrected_uniprot_accession"] != "P00734":
        raise ValueError("THRB correction differs")
    if registry["remaining_outcome_unseen_candidates_requiring_new_preregistration"] != 1 or registry["protocol_eligibility_established"]:
        raise ValueError("corrected candidate state differs")
    if any(value != 0 for value in result["data_boundary"].values()):
        raise ValueError("Stage112 crossed the protected-data boundary")
    decision = result["decision"]
    if not decision["new_thrombin_preregistration_required"]:
        raise ValueError("new preregistration was not required")
    if any(decision[key] for key in ("new_target_source_download_authorized", "new_coordinate_audit_authorized", "new_docking_authorized", "quantum_hardware_authorized")):
        raise ValueError("Stage112 prematurely released downstream work")
    return {
        "schema_version": "1.0",
        "status": "stage112_independent_audit_ok",
        "stage110_preserved": True,
        "corrected_target": "Thrombin (F2)",
        "next_stage_requires_new_preregistration": True,
        "later_compute_locked": True,
        "data_boundary": result["data_boundary"]
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage112_historical_candidate_pool_amendment01.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root, args.config)
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    path = root / config["outputs"]["audit_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
