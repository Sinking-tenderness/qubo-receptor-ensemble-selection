"""Independently audit Stage108's catalytic-domain No-Go decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    result = json.loads((root / config["outputs"]["summary_json"]).read_text(encoding="utf-8"))
    if result["status"] != "stage108_parp1_catalytic_domain_reference_feasibility_no_go":
        raise ValueError("Stage108 must record the frozen-pool No-Go")
    counts = result["counts"]
    if int(counts["metadata_eligible_count"]) != 10:
        raise ValueError("unexpected catalytic-domain candidate count")
    if int(counts["minimum_required_count"]) != 16 or bool(counts["pool_gate_passes"]):
        raise ValueError("pool gate must remain failed")
    if result["provisional_reference"]["pdb_id"] != "7KK5":
        raise ValueError("unexpected lowest-resolution provisional reference")
    if any(value != 0 for value in result["data_boundary"].values()):
        raise ValueError("Stage108 crossed the protected-data boundary")
    decision = result["decision"]
    if any(value for key, value in decision.items() if key.endswith("_authorized") or key.endswith("_released")):
        raise ValueError("Stage108 must not release downstream work")
    return {
        "schema_version": "1.0",
        "status": "stage108_independent_audit_ok",
        "metadata_eligible_count": int(counts["metadata_eligible_count"]),
        "minimum_required_count": int(counts["minimum_required_count"]),
        "pool_gate_passes": False,
        "provisional_reference": result["provisional_reference"]["pdb_id"],
        "later_stages_locked": True,
        "data_boundary": result["data_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage108_parp1_catalytic_domain_reference_feasibility.json"))
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
