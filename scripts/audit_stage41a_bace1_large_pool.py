"""Audit the Stage41a BACE1 large-pool freeze."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.freeze_stage41a_bace1_large_pool import (
    descriptor,
    file_sha256,
    portable_source_path,
    read_csv,
    read_json,
    rooted,
    verified,
    write_json,
)


def audit(config_path: Path, root: Path) -> dict:
    config = read_json(config_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    result = read_json(outputs["result_json"])
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    source_rows = read_csv(inputs["preparation_ready_pool"])
    frozen_rows = read_csv(outputs["large_pool_manifest_csv"])
    expected = int(config["pool_freeze"]["expected_preparation_ready_count"])
    checks = {
        "result_status": result.get("status") == "stage41a_bace1_large_pool_frozen",
        "config_identity": result["config"]["sha256"] == file_sha256(config_path),
        "implementation_identity": result["implementation"]["sha256"] == config["implementation"]["sha256"],
        "input_identities": all(result["inputs"][key]["sha256"] == value["sha256"] for key, value in config["inputs"].items()),
        "source_and_frozen_counts": len(source_rows) == len(frozen_rows) == expected,
        "unique_conformer_ids": len({row["conformer_id"] for row in frozen_rows}) == expected,
        "reference_first_once": frozen_rows[0]["pdb_id"] == config["target"]["reference_pdb_id"] and sum(row["is_reference"] == "True" for row in frozen_rows) == 1,
        "portable_relative_paths": all(not Path(row["mmcif_path"]).is_absolute() and not Path(row["aligned_protein_pdb_path"]).is_absolute() for row in frozen_rows),
        "no_structural_downselection": {row["conformer_id"] for row in source_rows} == {row["conformer_id"] for row in frozen_rows},
        "protected_boundaries_zero": all(int(result["data_boundary"][key]) == 0 for key in ("development_ligand_rows_read", "fresh_validation_rows_read", "locked_test_rows_read", "docking_scores_read", "new_docking_jobs", "quantum_hardware_jobs")),
        "redocking_gate_frozen": result["redocking_gate"] == config["redocking_gate"],
        "algorithmic_benchmark_frozen": result["algorithmic_benchmark"] == config["algorithmic_benchmark"],
    }
    source_by_id = {row["conformer_id"]: row for row in source_rows}
    file_checks: dict[str, bool] = {}
    for row in frozen_rows:
        source = source_by_id[row["conformer_id"]]
        mmcif = rooted(root, row["mmcif_path"])
        aligned = rooted(root, row["aligned_protein_pdb_path"])
        source_mmcif = portable_source_path(root, source["mmcif_path"])
        source_aligned = portable_source_path(root, source["aligned_protein_pdb_path"])
        file_checks[row["conformer_id"]] = (
            mmcif == source_mmcif
            and aligned == source_aligned
            and file_sha256(mmcif) == row["mmcif_sha256"] == source["mmcif_sha256"]
            and file_sha256(aligned) == row["aligned_protein_pdb_sha256"] == source["aligned_protein_pdb_sha256"]
        )
    checks["all_coordinate_file_identities"] = all(file_checks.values())
    maximum_size = int(config["algorithmic_benchmark"]["maximum_subset_size"])
    expected_states = {str(size): math.comb(expected, size) for size in range(1, maximum_size + 1)}
    checks["state_counts"] = result["counts"]["state_count_by_k"] == expected_states and int(result["counts"]["total_state_count_k1_to_k6"]) == sum(expected_states.values())
    checks["development_pair_count"] = int(config["development_ligand_protocol"]["maximum_pair_count_if_all_49_pass"]) == expected * int(config["development_ligand_protocol"]["total_ligand_count"]) * int(config["development_ligand_protocol"]["production_seed_count"])
    checks["output_hashes"] = all(result["outputs"][key]["sha256"] == file_sha256(outputs[key]) for key in result["outputs"])
    status = "stage41a_bace1_large_pool_freeze_audit_ok" if all(checks.values()) else "stage41a_bace1_large_pool_freeze_audit_failed"
    record = {
        "schema_version": "1.0",
        "status": status,
        "config": descriptor(root, config_path),
        "result": descriptor(root, outputs["result_json"]),
        "checks": checks,
        "coordinate_file_checks": file_checks,
        "recomputed_state_count_by_k": expected_states,
    }
    write_json(outputs["audit_json"], record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage41a_bace1_large_pool_freeze.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    record = audit(rooted(root, args.config), root)
    print(json.dumps({"status": record["status"], "checks": record["checks"]}, indent=2, sort_keys=True))
    return 0 if record["status"].endswith("_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
