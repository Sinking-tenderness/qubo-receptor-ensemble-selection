from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.experimental.unidock import prepare_stage42b_bace1_train266_inputs as common


def run(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config = common.read_json(config_path.resolve())
    outputs = dict(config["outputs"])
    summary_path = root / str(outputs["summary_json"])
    manifest_path = root / str(outputs["manifest_csv"])
    summary = common.read_json(summary_path)
    rows = common.read_csv(manifest_path)
    expected = dict(config["expected"])
    checks = {
        "preparation_status_ok": summary.get("status")
        == "stage91b_bace1_chembl365_unidock_inputs_ok",
        "manifest_hash_matches_summary": common.file_sha256(manifest_path)
        == str(summary["output"]["sha256"]).upper(),
        "ligand_count_matches": len(rows) == int(expected["ligand_count"]),
        "ligand_ids_unique": len({row["ligand_id"] for row in rows}) == len(rows),
        "development_only": {row["role"] for row in rows} == {"development"},
        "labels_match": Counter(row["potency_label"] for row in rows)
        == Counter(
            {
                key: int(value)
                for key, value in dict(expected["potency_label_counts"]).items()
            }
        ),
        "all_pdbqt_status_ok": all(row["pdbqt_status"] == "ok" for row in rows),
        "no_confirmation_or_test_rows": not any(
            row["role"] in {"confirmation_a", "confirmation_b", "locked_test"}
            for row in rows
        ),
        "no_closure_pseudoatoms": True,
        "all_file_hashes_match": True,
    }
    for row in rows:
        for path_key, hash_key in (
            ("sdf_path", "sdf_sha256"),
            ("pdbqt_path", "pdbqt_sha256"),
        ):
            path = root / row[path_key]
            if not path.is_file() or common.file_sha256(path) != row[hash_key].upper():
                checks["all_file_hashes_match"] = False
        if common.macrocycle_closure_atom_types(root / row["pdbqt_path"]):
            checks["no_closure_pseudoatoms"] = False
    audit = {
        "schema_version": "1.0",
        "status": (
            "stage91b_bace1_chembl365_input_independent_audit_ok"
            if all(checks.values())
            else "stage91b_bace1_chembl365_input_independent_audit_failed"
        ),
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "ligand_count": len(rows),
        "data_boundary": {
            "confirmation_rows_read": 0,
            "locked_test_rows_read": 0,
            "docking_scores_read": 0,
        },
        "development_docking_release": all(checks.values()),
    }
    audit_path = root / str(outputs["audit_json"])
    common.write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["failed_checks"]:
        raise SystemExit(1)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage91b_bace1_chembl365_unidock_input_preparation.json"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
