"""Independently audit the ESR2 source-reference No-Go gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    result = json.loads((root / config["outputs"]["summary_json"]).read_text(encoding="utf-8"))
    if result["status"] != "stage109_esr2_reference_identity_no_go":
        raise ValueError("Stage109 must be a No-Go")
    if result["ranking_provenance"]["historical_eligible_order"] != ["PPARA", "PPARD", "ESR2"]:
        raise ValueError("historical target order differs")
    reference = result["reference_metadata"]
    if reference["pdb_id"] != "2FSZ" or reference["mutation_count"] != 3:
        raise ValueError("DUD-E reference identity differs")
    if reference["pdbx_mutation_note"] != "C334S, C369S, C481S":
        raise ValueError("ESR2 mutation record differs")
    checks = result["gate_checks"]
    if checks["reference_is_wild_type"] or checks["reference_is_metadata_eligible"] or not checks["receptor_pool_passes"] or checks["gate_passes"]:
        raise ValueError("ESR2 reference gate logic differs")
    if result["counts"]["metadata_eligible_receptor_count"] != 32:
        raise ValueError("ESR2 metadata pool count differs")
    boundary = result["data_boundary"]
    if boundary["source_active_lines_counted"] != 367 or boundary["source_decoy_lines_counted"] != 20199:
        raise ValueError("ESR2 source line counts differ")
    if boundary["source_label_values_parsed_or_used"] != 0:
        raise ValueError("Stage109 must not parse or use source label values")
    if any(value != 0 for key, value in boundary.items() if key not in {"source_active_lines_counted", "source_decoy_lines_counted"}):
        raise ValueError("Stage109 crossed its data boundary")
    if any(value for key, value in result["decision"].items() if key.endswith("_authorized")):
        raise ValueError("Stage109 released a downstream task")
    return {"schema_version": "1.0", "status": "stage109_independent_audit_ok", "dude_reference": "2FSZ", "reference_mutation_count": 3, "metadata_eligible_receptor_count": 32, "later_stages_locked": True, "data_boundary": result["data_boundary"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage109_esr2_reference_identity_gate.json"))
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
