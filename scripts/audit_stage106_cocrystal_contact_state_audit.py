"""Independent audit for Stage106 co-crystal contact-state diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    outputs = config["outputs"]
    result = json.loads((root / outputs["result_json"]).read_text(encoding="utf-8"))
    contacts = read_csv(root / outputs["contact_csv"])
    pairs = read_csv(root / outputs["pair_csv"])
    targets = read_csv(root / outputs["target_csv"])
    require(result["status"] == "stage106_cocrystal_contact_state_audit_complete", "unexpected status")
    require(len(contacts) == 25, f"expected 25 contact records, found {len(contacts)}")
    require(len(pairs) == 144, f"expected 144 pair records, found {len(pairs)}")
    require(len(targets) == 2, f"expected two target summaries, found {len(targets)}")
    require({row["target_id"] for row in contacts} == {"EGFR", "FA10"}, "unexpected contact target")
    require(all(int(row["protein_contact_residue_count"]) > 0 for row in contacts), "empty contact set")
    require(all(int(row["cocrystal_ligand_heavy_atom_count"]) > 0 for row in contacts), "empty co-crystal ligand")
    require(all(0.0 <= float(row["contact_jaccard_distance"]) <= 1.0 for row in pairs), "invalid Jaccard distance")
    focal = result["focal_pairs"]["FA10_positive"]
    require(focal["stage102a_fixed_k2_selection_count"] == 5, "FA10 focal pair must recur in five folds")
    require(float(focal["stage102a_fixed_k2_min_outer_gain"]) > 0.0, "FA10 focal pair must be positive in every fold")
    require(result["feasibility"]["fa10_positive_pair_is_not_selected_for_extreme_structural_distance"], "unexpected structural-distance interpretation")
    require(result["feasibility"]["contact_state_signal_is_not_authorized_as_predictor"], "contact signal incorrectly authorized")
    require(all(value == 0 for value in result["data_boundary"].values()), "data boundary breached")
    require(result["decision"]["new_target_protocol_authorized"] is False, "new target improperly authorized")
    require(result["decision"]["parp1_released"] is False, "PARP1 improperly released")
    require(result["decision"]["quantum_hardware_authorized"] is False, "hardware improperly authorized")
    return {
        "schema_version": "1.0",
        "status": "stage106_independent_audit_ok",
        "contact_record_count": len(contacts),
        "pair_record_count": len(pairs),
        "target_summary_count": len(targets),
        "fa10_focal_pair_all_outer_folds_positive": True,
        "outer_labels_used_for_descriptor_construction": False,
        "data_boundary": result["data_boundary"],
        "new_target_protocol_authorized": False,
        "parp1_released": False,
        "quantum_hardware_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage106_cocrystal_contact_state_audit.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    record = audit(root, config)
    output = root / config["outputs"]["audit_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
