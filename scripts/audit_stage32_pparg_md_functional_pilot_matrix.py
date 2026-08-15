"""Independently audit the completed Stage32 PPARG MD pilot matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {"path": path.resolve().relative_to(root.resolve()).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    inputs = {key: root / value for key, value in config["inputs"].items()}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = read_json(inputs["summary_json"])
    progress = read_json(inputs["progress_json"])
    preparation = read_json(inputs["input_preparation_result"])
    if summary.get("status") != "stage32_pparg_md_functional_pilot_matrix_ok" or progress.get("status") != "stage32_production_complete" or preparation.get("status") != "stage32_inputs_ok":
        raise ValueError("Stage32 completion status differs")
    if sha256(inputs["executed_config"]) != summary["config"]["sha256"]:
        raise ValueError("Stage32 executed config hash differs")
    for record in summary["outputs"].values():
        path = root / record["path"]
        if sha256(path) != record["sha256"] or path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Stage32 output descriptor differs: {record['path']}")

    receptors = read_csv(inputs["prepared_receptor_manifest"])
    ligands = read_csv(inputs["ligand_manifest"])
    scores = read_csv(inputs["scores_csv"])
    batches = read_csv(inputs["batch_runs_csv"])
    median_rows = read_csv(inputs["median_matrix_csv"])
    minimum_rows = read_csv(inputs["minimum_matrix_csv"])
    receptor_ids = [row["conformer_id"] for row in receptors]
    ligand_ids = [row["ligand_id"] for row in ligands]
    gate = config["technical_gate"]
    if len(receptor_ids) != int(gate["required_receptor_count"]) or len(set(receptor_ids)) != len(receptor_ids):
        raise ValueError("Stage32 receptor coverage differs")
    if len(ligand_ids) != int(gate["required_ligand_count"]) or len(set(ligand_ids)) != len(ligand_ids):
        raise ValueError("Stage32 ligand coverage differs")
    if Counter(row["label"] for row in ligands) != Counter({"active": 80, "decoy": 80}):
        raise ValueError("Stage32 label balance differs")
    if len(batches) != 48 or len(scores) != 7680:
        raise ValueError("Stage32 batch or pair count differs")
    seeds = ("seed0", "seed1", "seed2")
    expected_batch_keys = {(seed, receptor) for seed in seeds for receptor in receptor_ids}
    batch_keys = {(row["seed_id"], row["receptor_id"]) for row in batches}
    if batch_keys != expected_batch_keys or any(row["status"] != "ok" for row in batches):
        raise ValueError("Stage32 batch coverage or status differs")
    expected_score_keys = {(seed, receptor, ligand) for seed in seeds for receptor in receptor_ids for ligand in ligand_ids}
    score_keys = {(row["seed_id"], row["receptor_id"], row["ligand_id"]) for row in scores}
    if score_keys != expected_score_keys:
        raise ValueError("Stage32 score key coverage differs")
    if any(row["status"] != "ok" or row["pose_integrity_status"] != "ok" or row["atom_count_match"].lower() != "true" or row["atom_types_match"].lower() != "true" or row["single_pose_match"].lower() != "true" for row in scores):
        raise ValueError("Stage32 score technical status differs")
    values: dict[tuple[str, str], list[float]] = {}
    for row in scores:
        score = float(row["gpu_score"])
        if not math.isfinite(score) or abs(score) > 1000:
            raise ValueError("Stage32 score is invalid")
        values.setdefault((row["ligand_id"], row["receptor_id"]), []).append(score)
    median_by_ligand = {row["ligand_id"]: row for row in median_rows}
    minimum_by_ligand = {row["ligand_id"]: row for row in minimum_rows}
    if set(median_by_ligand) != set(ligand_ids) or set(minimum_by_ligand) != set(ligand_ids):
        raise ValueError("Stage32 matrix ligand coverage differs")
    maximum_difference = 0.0
    for ligand in ligand_ids:
        for receptor in receptor_ids:
            local = values[(ligand, receptor)]
            if len(local) != 3:
                raise ValueError("Stage32 seed replicate count differs")
            maximum_difference = max(
                maximum_difference,
                abs(statistics.median(local) - float(median_by_ligand[ligand][receptor])),
                abs(min(local) - float(minimum_by_ligand[ligand][receptor])),
            )
    known_warnings = sum(int(row["known_warning_event_count"]) for row in batches)
    unresolved = sum(int(row["unresolved_warning_event_count"]) for row in batches)
    pose_failures = sum(int(row["pose_integrity_failure_count"]) for row in batches)
    data_boundary_zero = int(summary["data_boundary"]["fresh_validation_rows_read"]) == 0 and int(summary["data_boundary"]["test_rows_read"]) == 0
    checks = {
        "source_package_manifest_verified": int(config["source_package"]["manifest_verification_failures"]) == 0,
        "executed_config_hash_verified": True,
        "all_summary_output_descriptors_verified": True,
        "all_48_batches_complete_and_unique": True,
        "all_7680_score_keys_complete_and_unique": True,
        "all_pose_integrity_fields_pass": True,
        "median_and_minimum_matrices_recomputed": maximum_difference <= float(gate["maximum_matrix_recomputation_abs_difference"]),
        "unresolved_warning_gate_passed": unresolved <= int(gate["maximum_unresolved_warning_count"]),
        "pose_integrity_gate_passed": pose_failures <= int(gate["maximum_pose_integrity_failure_count"]),
        "protected_data_boundary_zero": data_boundary_zero,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage32 matrix audit failed: {checks}")
    result = {
        "schema_version": "1.0",
        "status": "stage32_pparg_md_functional_pilot_matrix_audit_ok",
        "config": descriptor(root, config_path),
        "checks": checks,
        "coverage": {"receptor_count": len(receptors), "ligand_count": len(ligands), "seed_count": 3, "batch_count": len(batches), "pair_count": len(scores), "known_warning_event_count": known_warnings, "unresolved_warning_event_count": unresolved, "pose_integrity_failure_count": pose_failures, "maximum_matrix_recomputation_abs_difference": maximum_difference},
        "inputs": {key: descriptor(root, path) for key, path in inputs.items()},
        "data_boundary": {"train_rows_read": 160, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output = root / config["outputs"]["matrix_audit_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32a_pparg_md_functional_landscape_analysis.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    audit(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
