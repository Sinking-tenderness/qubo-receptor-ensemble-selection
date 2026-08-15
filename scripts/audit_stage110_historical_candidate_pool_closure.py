"""Independently audit closure of the finite historical target registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(root: Path, config_path: Path) -> dict:
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    result = json.loads((root / config["outputs"]["summary_json"]).read_text(encoding="utf-8"))
    if result["status"] != "stage110_historical_candidate_pool_closed":
        raise ValueError("Stage110 status differs")
    if result["counts"]["remaining_outcome_unseen_protocol_eligible_candidates"] != 0:
        raise ValueError("historical registry should have no remaining candidate")
    entries = {entry["target_id"]: entry for entry in result["registry_entries"]}
    if entries["ESR2"]["status"] != "reference No-Go" or entries["PARP1"]["status"] != "historical exploratory No-Go":
        raise ValueError("new closure records differ")
    if entries["PPARD"]["status"] != "already used and closed":
        raise ValueError("PPARD closure differs")
    if any(value != 0 for value in result["data_boundary"].values()):
        raise ValueError("Stage110 accessed new target data")
    if any(value for key, value in result["decision"].items() if key.endswith("_authorized")):
        raise ValueError("Stage110 improperly released later work")
    return {"schema_version": "1.0", "status": "stage110_independent_audit_ok", "historical_candidate_pool_closed": True, "remaining_outcome_unseen_protocol_eligible_candidates": 0, "later_stages_locked": True, "data_boundary": result["data_boundary"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage110_historical_candidate_pool_closure.json"))
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
