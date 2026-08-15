"""Independently audit the final Stage 08c replacement and receptor manifest."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

try:
    from scripts.evaluate_redocking_rmsd import (
        calculate_pose_rmsds,
        parse_vina_affinities,
    )
    from scripts.prepare_receptor import file_sha256
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.evaluate_redocking_rmsd import (
        calculate_pose_rmsds,
        parse_vina_affinities,
    )
    from scripts.prepare_receptor import file_sha256


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def rooted(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path leaves audit root: {value}") from error
    return path


def checked_output(root: Path, descriptor: dict[str, object]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    if not path.is_file() or file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"output identity differs: {path}")
    return path


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1"}


def run_audit(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    outputs = dict(config["outputs"])
    summary_path = rooted(root, str(outputs["summary_json"]))
    summary = read_json(summary_path)
    if summary.get("status") != "stage08c_final_replacement_redocking_gate_ok":
        raise ValueError("Stage 08c source gate did not pass")
    if str(summary["config"]["sha256"]).upper() != file_sha256(config_path):
        raise ValueError("Stage 08c source config hash differs")
    boundary = dict(summary["data_boundary"])
    if any(int(value) != 0 for value in boundary.values()):
        raise ValueError("Stage 08c source crossed a data boundary")
    source_outputs = dict(summary["outputs"])
    results_path = checked_output(root, dict(source_outputs["redocking_results_csv"]))
    final_path = checked_output(root, dict(source_outputs["final_receptor_manifest_csv"]))
    rows = read_csv(results_path)
    final_rows = read_csv(final_path)
    expected = dict(config["expected"])
    if len(rows) != int(expected["redocking_pair_count"]):
        raise ValueError("Stage 08c result count differs")
    if len(final_rows) != int(expected["final_receptor_count"]):
        raise ValueError("Stage 08c final receptor count differs")
    expected_ids = [
        *[str(value) for value in expected["current_receptor_ids"]],
        str(expected["replacement_receptor_id"]),
    ]
    final_ids = [row["conformer_id"] for row in final_rows]
    if final_ids != expected_ids or len(set(final_ids)) != len(final_ids):
        raise ValueError("Stage 08c final receptor order differs")
    for row in final_rows:
        receptor = rooted(root, row["receptor_pdbqt"])
        if row["status"] != "ok" or file_sha256(receptor) != row[
            "receptor_pdbqt_sha256"
        ].upper():
            raise ValueError(f"final receptor identity differs: {row['conformer_id']}")
    seeds = [str(row["seed_id"]) for row in dict(config["inputs"])["seeds"]]
    receptor_id = str(expected["replacement_receptor_id"])
    if {(row["conformer_id"], row["seed_id"]) for row in rows} != {
        (receptor_id, seed_id) for seed_id in seeds
    }:
        raise ValueError("Stage 08c receptor/seed grid differs")
    threshold = float(dict(config["redocking_gate"])["maximum_rmsd_angstrom"])
    recomputed: list[dict[str, object]] = []
    for row in rows:
        reference = rooted(root, row["reference_sdf"])
        docked = rooted(root, row["docked_pdbqt"])
        if file_sha256(reference) != row["reference_sdf_sha256"].upper():
            raise ValueError("Stage 08c reference hash differs")
        if file_sha256(docked) != row["docked_pdbqt_sha256"].upper():
            raise ValueError("Stage 08c pose hash differs")
        affinities = parse_vina_affinities(
            docked.read_text(encoding="ascii", errors="replace")
        )
        rmsds = calculate_pose_rmsds(reference, docked)
        if len(affinities) != 1 or len(rmsds) != 1:
            raise ValueError("Stage 08c output must contain one pose")
        if abs(affinities[0] - float(row["top_ranked_affinity_kcal_per_mol"])) > 1e-9:
            raise ValueError("Stage 08c affinity differs")
        if abs(rmsds[0] - float(row["top_ranked_rmsd_angstrom"])) > 1e-6:
            raise ValueError("Stage 08c RMSD differs")
        success = rmsds[0] <= threshold
        if success != parse_bool(row["top_ranked_pose_success"]):
            raise ValueError("Stage 08c success flag differs")
        if int(row["unresolved_warning_event_count"]) != 0 or int(
            row["pose_integrity_failure_count"]
        ) != 0:
            raise ValueError("Stage 08c technical integrity differs")
        recomputed.append(
            {
                "seed_id": row["seed_id"],
                "rmsd_angstrom": rmsds[0],
                "success": success,
            }
        )
    rmsd_values = [float(row["rmsd_angstrom"]) for row in recomputed]
    successful = sum(bool(row["success"]) for row in recomputed)
    median_rmsd = statistics.median(rmsd_values)
    minimum_successes = int(
        dict(config["redocking_gate"])["minimum_successful_seeds_per_receptor"]
    )
    passed = successful >= minimum_successes and median_rmsd <= threshold
    if not passed:
        raise ValueError("independent Stage 08c gate failed")
    result = {
        "schema_version": "1.0",
        "audit_id": "stage08c-mk14-final-replacement-independent-audit-v1",
        "status": "independent_stage08c_final_replacement_audit_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_summary": {
            "path": summary_path.relative_to(root).as_posix(),
            "sha256": file_sha256(summary_path),
        },
        "replacement_receptor_id": receptor_id,
        "successful_seed_count": successful,
        "median_rmsd_angstrom": median_rmsd,
        "maximum_rmsd_angstrom": max(rmsd_values),
        "final_receptor_count": len(final_rows),
        "final_receptor_ids": final_ids,
        "unresolved_warning_event_count": 0,
        "pose_integrity_failure_count": 0,
        "data_boundary": {str(key): int(value) for key, value in boundary.items()},
        "next_gate": "preregister the Train-696 x 16 receptors x 3 seeds Uni-Dock production matrix",
        "decision_boundary": "This audit establishes only the final receptor input pool and cognate-pose admission. It does not establish enrichment, QUBO benefit, or quantum advantage.",
    }
    output_path = rooted(root, str(outputs["audit_json"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_audit(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
