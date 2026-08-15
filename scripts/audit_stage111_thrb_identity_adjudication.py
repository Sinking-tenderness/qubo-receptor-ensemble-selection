"""Independently audit the historical THRB identity correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(root: Path, config_path: Path) -> dict:
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    result = json.loads((root / config["outputs"]["summary_json"]).read_text(encoding="utf-8"))
    if result["status"] != "stage111_thrb_identity_mismatch_confirmed":
        raise ValueError("identity mismatch was not recorded")
    historical = result["historical_mapping"]
    authoritative = result["authoritative_identity"]
    if historical["uniprot_accession"] != "P10828" or authoritative["rcsb_uniprot_accession"] != "P00734":
        raise ValueError("source identities do not differ as expected")
    if authoritative["dude_catalog_description"] != "Thrombin" or authoritative["rcsb_reference_pdb"] != "1YPE":
        raise ValueError("thrombin evidence differs")
    if any(value != 0 for value in result["data_boundary"].values()):
        raise ValueError("Stage111 crossed the protected-data boundary")
    decision = result["decision"]
    if decision["historical_record_overwritten"] or not decision["stage110_registry_requires_amendment"] or not decision["thrombin_new_preregistration_authorized"]:
        raise ValueError("correction decision differs")
    if any(decision[key] for key in ("thrombin_source_download_authorized", "thrombin_coordinate_download_authorized", "thrombin_docking_authorized", "quantum_hardware_authorized")):
        raise ValueError("Stage111 prematurely released compute work")
    return {"schema_version": "1.0", "status": "stage111_independent_audit_ok", "historical_record_overwritten": False, "corrected_dude_target": "Thrombin", "corrected_uniprot_accession": "P00734", "later_compute_locked": True, "data_boundary": result["data_boundary"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage111_thrb_identity_adjudication.json"))
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
